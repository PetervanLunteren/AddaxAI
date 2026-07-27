"""Tests for `app.db.migrations` and the init_db reconciliation flow.

The legacy stamp_head() path stamped some beta-tester DBs at head
without their schema actually being at head. The fingerprint-based
detection in `reconcile_alembic_version` is the recovery and the
guarantee for the future, so the assertions here are: every alembic
revision on disk has a fingerprint entry, the fingerprint walk picks
the right revision for each known schema shape, and `init_db()` of a
DB stamped at the wrong revision repairs itself on the next startup.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from app.core.config import Settings
from app.db.base import init_db
from app.db.migrations import (
    SCHEMA_FINGERPRINTS,
    detect_schema_revision,
    get_current_revision,
    get_head_revision,
    reconcile_alembic_version,
    upgrade_to_head,
)


@pytest.fixture()
def isolated_db_settings(tmp_path: Path, monkeypatch):
    """Point get_settings() at a fresh empty user-data dir.

    Each test gets its own SQLite file so init_db() can run end-to-end
    without colliding with the developer's real `~/AddaxAI/addaxai.db`.
    """
    db_path = tmp_path / "addaxai.db"
    settings = Settings(
        user_data_dir=tmp_path,
        database_url=f"sqlite:///{db_path}",
    )

    def _get_settings() -> Settings:
        return settings

    monkeypatch.setattr("app.core.config.get_settings", _get_settings)
    monkeypatch.setattr("app.db.base.get_settings", _get_settings)
    monkeypatch.setattr("app.db.migrations.get_settings", _get_settings)

    # Reset the cached engine so tests don't reuse one bound to another
    # tmp_path's DB file.
    from app.db import base as base_mod
    if hasattr(base_mod, "_cached_engine"):
        base_mod._cached_engine = None

    yield settings


def _engine_for(settings: Settings):
    """A fresh SQLAlchemy engine pointed at the test DB."""
    return create_engine(settings.database_url, future=True)


# ---------------------------------------------------------------------------
# Static guarantees about the fingerprint table itself
# ---------------------------------------------------------------------------


def test_every_alembic_revision_has_a_fingerprint() -> None:
    """Every migration script on disk must have a SCHEMA_FINGERPRINTS row.

    Forgetting to add the row is the bug we are guarding against, so
    catching it here is the cheapest place. If this test fails, append
    an entry to SCHEMA_FINGERPRINTS for the new revision picking a
    column the migration adds (or table=created_table, column=None for
    schema bootstraps).
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    backend_dir = Path(__file__).resolve().parents[2]
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    script_dir = ScriptDirectory.from_config(cfg)

    on_disk = {rev.revision for rev in script_dir.walk_revisions()}
    in_table = {fp.revision for fp in SCHEMA_FINGERPRINTS}
    missing = on_disk - in_table

    assert not missing, (
        f"Migrations without SCHEMA_FINGERPRINTS entries: {sorted(missing)}. "
        f"Add one row per missing revision (see docstring of "
        f"app.db.migrations)."
    )


def test_fingerprints_are_in_chronological_order() -> None:
    """SCHEMA_FINGERPRINTS must list revisions oldest -> newest.

    detect_schema_revision walks the list in reverse and returns the
    first hit, so out-of-order rows silently break detection.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    backend_dir = Path(__file__).resolve().parents[2]
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    script_dir = ScriptDirectory.from_config(cfg)

    # walk_revisions yields newest-first; reverse for oldest-first.
    chronological = list(reversed(list(script_dir.walk_revisions())))
    expected_order = [r.revision for r in chronological]
    actual_order = [fp.revision for fp in SCHEMA_FINGERPRINTS]

    assert actual_order == expected_order, (
        "SCHEMA_FINGERPRINTS is out of chronological order. "
        f"Expected: {expected_order}. Got: {actual_order}."
    )


# ---------------------------------------------------------------------------
# detect_schema_revision against synthetic schema shapes
# ---------------------------------------------------------------------------


def test_detect_returns_none_for_empty_db(isolated_db_settings) -> None:
    """No user tables → returns None (fresh install signal)."""
    engine = _engine_for(isolated_db_settings)
    assert detect_schema_revision(engine) is None


def test_detect_returns_initial_for_initial_schema(isolated_db_settings) -> None:
    """Floor marker `files.captured_at_local` present, no later columns → initial."""
    engine = _engine_for(isolated_db_settings)
    with engine.begin() as conn:
        # The initial-schema fingerprint is files.captured_at_local. We
        # don't need the full table here; the fingerprint only checks for
        # the table / column it knows about, and none of the newer
        # detectable columns exist, so the walk falls through to initial.
        conn.execute(
            text("CREATE TABLE files (id TEXT PRIMARY KEY, captured_at_local TEXT)")
        )

    assert detect_schema_revision(engine) == "9c173fff3bcd"


def test_detect_returns_warnings_revision_when_warnings_column_exists(
    isolated_db_settings,
) -> None:
    """`deployments.warnings` present → latest fingerprinted revision."""
    engine = _engine_for(isolated_db_settings)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE projects (id TEXT PRIMARY KEY)"))
        conn.execute(
            text(
                "CREATE TABLE deployments ("
                "id TEXT PRIMARY KEY, "
                "warnings TEXT"
                ")"
            )
        )

    assert detect_schema_revision(engine) == "2540e6edbee2"


# ---------------------------------------------------------------------------
# init_db reconciliation: end-to-end repair of a Cara-shaped DB
# ---------------------------------------------------------------------------


def test_init_db_repairs_db_stamped_at_wrong_revision(
    isolated_db_settings,
) -> None:
    """Reproduce Cara's state and verify init_db repairs it.

    Cara's DB had `alembic_version = 2540e6edbee2` (the head migration
    revision) but the actual schema was missing the two columns those
    migrations add: `projects.observations_max_detections` and
    `deployments.warnings`. Every query that touched either column
    crashed with `sqlite3.OperationalError: no such column`.

    init_db must detect the mismatch, re-stamp at the actual revision
    the schema corresponds to, and run upgrade_to_head to add the
    missing columns. After this, the live schema must contain both
    columns and `alembic_version` must reflect head.
    """
    # 1) Build a DB at the initial schema (9c173fff3bcd) and stamp at
    #    head. This is exactly the state the buggy legacy stamp_head()
    #    code produced.
    upgrade_to_head()  # creates the full schema at head first
    engine = _engine_for(isolated_db_settings)

    head = get_head_revision()

    # Drop every column that post-initial migrations add so the live
    # schema LOOKS like initial schema but is stamped at head. SQLite
    # supports DROP COLUMN since 3.35. The list grows as we add
    # column-introducing migrations; keep it in sync with
    # SCHEMA_FINGERPRINTS' detectable entries beyond the initial one.
    with engine.begin() as conn:
        # observations_max_detections was added by revision 03e058c707df
        # and later dropped by d4e5f6a7b8c9, so it is not in the head
        # schema and is not part of the "looks like initial" simulation.
        conn.execute(text("ALTER TABLE deployments DROP COLUMN warnings"))
        conn.execute(text("DROP INDEX IF EXISTS ix_projects_mode"))
        conn.execute(text("ALTER TABLE projects DROP COLUMN mode"))
        conn.execute(text("ALTER TABLE projects DROP COLUMN folder_run_state"))
        # common_name (detections) was added by c9d0e1f2a3b4 and
        # suggestion_dismissed by d0e1f2a3b4c5. Drop both so the schema
        # truly looks initial and the fingerprint walk does not stop there.
        conn.execute(text("ALTER TABLE detections DROP COLUMN common_name"))
        conn.execute(
            text("ALTER TABLE detections DROP COLUMN suggestion_dismissed")
        )
        # events.confirmed (detectable, renamed from verified by a3b4c5d6e7f8)
        # + event_observations.human_count are the newest detectable schema.
        conn.execute(text("ALTER TABLE events DROP COLUMN confirmed"))
        conn.execute(
            text("ALTER TABLE event_observations DROP COLUMN human_count")
        )
        # files.frames_processed was added by b4c5d6e7f8a9.
        conn.execute(text("ALTER TABLE files DROP COLUMN frames_processed"))
        # projects.classification_gate was added by c5d6e7f8a9b0.
        conn.execute(
            text("ALTER TABLE projects DROP COLUMN classification_gate")
        )
        # deployments.classification_gate_used was added by d6e7f8a9b0c1.
        conn.execute(
            text(
                "ALTER TABLE deployments DROP COLUMN classification_gate_used"
            )
        )
        # projects.media_filter is the newest detectable schema
        # (2b3c4d5e6f7a, which media a new analysis reads off disk), with
        # detection_augment + detection_image_size (1a2b3c4d5e6f, advanced
        # MegaDetector inference options) behind it. Drop all three so the
        # fingerprint walk doesn't stop at head.
        conn.execute(text("ALTER TABLE projects DROP COLUMN media_filter"))
        conn.execute(
            text("ALTER TABLE projects DROP COLUMN detection_augment")
        )
        conn.execute(
            text("ALTER TABLE projects DROP COLUMN detection_image_size")
        )
        # projects.counting_threshold is also detectable (f8a9b0c1d2e3
        # renamed it from detection_threshold). Rename it back to the
        # original name so the schema truly looks initial and the
        # fingerprint walk doesn't stop at that revision.
        conn.execute(
            text(
                "ALTER TABLE projects "
                "RENAME COLUMN counting_threshold TO detection_threshold"
            )
        )

    # alembic_version row stays at head — that is exactly the bug.
    assert get_current_revision(engine) == head

    # Sanity-check: the schema looks like initial, the version row lies.
    insp = inspect(engine)
    deployments_cols = {c["name"] for c in insp.get_columns("deployments")}
    assert "warnings" not in deployments_cols

    # 2) Run init_db. Reconciliation should re-stamp at the initial
    #    revision and then upgrade_to_head should re-add both columns.
    init_db()

    # 3) Every dropped column is back, alembic_version is at head.
    insp = inspect(engine)
    projects_cols = {c["name"] for c in insp.get_columns("projects")}
    deployments_cols = {c["name"] for c in insp.get_columns("deployments")}
    assert "warnings" in deployments_cols
    assert "mode" in projects_cols
    assert "folder_run_state" in projects_cols
    # The rename was re-applied: new name present, old name gone.
    assert "counting_threshold" in projects_cols
    assert "detection_threshold" not in projects_cols
    assert get_current_revision(engine) == head


def test_init_db_noop_on_already_healthy_db(isolated_db_settings) -> None:
    """A DB already at head must reach the second init_db unchanged."""
    init_db()  # first init_db: creates fresh schema from scratch
    engine = _engine_for(isolated_db_settings)
    head = get_head_revision()
    assert get_current_revision(engine) == head

    init_db()  # second init_db: must be a no-op
    assert get_current_revision(engine) == head


def test_reconcile_returns_none_for_fresh_install(isolated_db_settings) -> None:
    """Empty DB → reconcile_alembic_version is a no-op signalling fresh."""
    engine = _engine_for(isolated_db_settings)
    assert reconcile_alembic_version(engine) is None
    # No alembic_version row created.
    assert get_current_revision(engine) is None


def test_reconcile_refuses_unknown_stamped_revision(isolated_db_settings) -> None:
    """A DB stamped at a revision missing from SCHEMA_FINGERPRINTS must fail loud.

    Replays the 2026-05-27 incident: a new migration was added on disk
    without a corresponding SCHEMA_FINGERPRINTS entry. The old reconcile
    silently re-stamped backward and re-ran the chain, and the rebuild
    destroyed user data. The reconciler must now refuse and surface a
    clear error instead of auto-resolving.
    """
    upgrade_to_head()
    engine = _engine_for(isolated_db_settings)

    # Stamp at a revision string that does NOT appear in SCHEMA_FINGERPRINTS.
    fake_revision = "ffffffffffff"
    assert not any(fp.revision == fake_revision for fp in SCHEMA_FINGERPRINTS)
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = :v"), {"v": fake_revision})

    with pytest.raises(RuntimeError, match=r"not registered in SCHEMA_FINGERPRINTS"):
        reconcile_alembic_version(engine)

    # alembic_version unchanged — the gate refuses without writing.
    assert get_current_revision(engine) == fake_revision


def test_reconcile_stamps_legacy_db_without_alembic_version(
    isolated_db_settings,
) -> None:
    """User tables present at the floor but no alembic_version → stamp at detected rev."""
    engine = _engine_for(isolated_db_settings)
    with engine.begin() as conn:
        # Floor schema: the initial-schema marker column is present, so
        # the DB is recognised as at-least-initial and stamped there.
        conn.execute(
            text("CREATE TABLE files (id TEXT PRIMARY KEY, captured_at_local TEXT)")
        )

    # No alembic_version yet.
    assert get_current_revision(engine) is None

    result = reconcile_alembic_version(engine)
    assert result == "9c173fff3bcd"
    assert get_current_revision(engine) == "9c173fff3bcd"


def test_reconcile_refuses_pre_floor_db(isolated_db_settings) -> None:
    """A DB older than the floor must fail loud instead of crashing mid-upgrade.

    Reproduces issue #11 (Arky's Linux install): a legacy beta DB with
    user tables but whose `files` table predates `captured_at_local`.
    The old code stamped it at the initial revision on the strength of a
    `projects` table existing, then the forward chain died with
    `KeyError: 'captured_at_local'` in the nullable-capture-dates
    migration. Reconcile must now refuse up front and leave the DB
    untouched.
    """
    engine = _engine_for(isolated_db_settings)
    with engine.begin() as conn:
        # User tables exist, but none carry the floor marker column
        # (files.captured_at_local). This is a pre-floor schema.
        conn.execute(text("CREATE TABLE projects (id TEXT PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE files (id TEXT PRIMARY KEY, captured_at TEXT)"))

    assert get_current_revision(engine) is None

    with pytest.raises(RuntimeError, match=r"older than the oldest supported schema"):
        reconcile_alembic_version(engine)

    # Left untouched: no alembic_version row written.
    assert get_current_revision(engine) is None


def test_upgrade_from_base_matches_models(isolated_db_settings) -> None:
    """Running the whole chain from base must produce the schema the models expect.

    This is the immutability guard: it catches a future migration that
    drifts from `Base.metadata` (references a column the chain never
    creates, or forgets to add one), which is the class of bug that
    produced issue #11. If this fails, the migrations and the ORM models
    disagree and the app would crash at runtime with "no such column".
    """
    import app.models  # noqa: F401  # populates Base.metadata
    from app.db.base import Base

    upgrade_to_head()
    engine = _engine_for(isolated_db_settings)
    insp = inspect(engine)

    live_tables = set(insp.get_table_names())
    for table_name, table in Base.metadata.tables.items():
        assert table_name in live_tables, (
            f"Model table {table_name!r} is missing after upgrade to head. "
            f"A migration is out of sync with the ORM models."
        )
        live_columns = {c["name"] for c in insp.get_columns(table_name)}
        model_columns = {c.name for c in table.columns}
        missing = model_columns - live_columns
        assert not missing, (
            f"Table {table_name!r} is missing columns {sorted(missing)} after "
            f"upgrade to head. A migration is out of sync with the ORM models."
        )


def test_every_detectable_fingerprint_is_satisfied_at_head(
    isolated_db_settings,
) -> None:
    """A fingerprint column must still exist once the chain reaches head.

    A fingerprint says "the schema is at this revision or later when this
    column exists". If a later migration drops or renames that column, the
    fingerprint can never be satisfied again, and
    `_alembic_version_is_truthful` then returns False on every launch for
    every user: the DB gets re-stamped backwards and the tail of the chain
    re-runs on each start, taking an unpruned pre-upgrade backup each time.

    This is not hypothetical. `f2a3b4c5d6e7` was fingerprinted on
    `events.verified`, which the very next revision renames to
    `events.confirmed`.
    """
    from app.db.migrations import _fingerprint_satisfied

    upgrade_to_head()
    engine = _engine_for(isolated_db_settings)

    broken = [
        f"{fp.revision} -> {fp.table}.{fp.column}"
        for fp in SCHEMA_FINGERPRINTS
        if fp.is_detectable and not _fingerprint_satisfied(fp, engine)
    ]
    assert not broken, (
        f"These fingerprints point at schema that no longer exists at head: "
        f"{broken}. Repoint each at a column its own migration adds and that "
        f"survives to head, or make the entry non-detectable."
    )


def test_upgrade_from_base_creates_every_model_index(isolated_db_settings) -> None:
    """Every index declared on a model must exist after upgrading to head.

    `test_upgrade_from_base_matches_models` only compares tables and columns,
    so adding an `Index(...)` to a model without a matching migration used to
    pass CI while doing nothing on real databases. That matters most for
    indexes nobody queries through: the ones covering a foreign key's child
    column, whose absence only shows up as a delete that takes hours.
    """
    import app.models  # noqa: F401  # populates Base.metadata
    from app.db.base import Base

    upgrade_to_head()
    insp = inspect(_engine_for(isolated_db_settings))

    for table_name, table in Base.metadata.tables.items():
        model_indexes = {idx.name for idx in table.indexes}
        if not model_indexes:
            continue
        live_indexes = {i["name"] for i in insp.get_indexes(table_name)}
        missing = model_indexes - live_indexes
        assert not missing, (
            f"Table {table_name!r} is missing indexes {sorted(missing)} after "
            f"upgrade to head. Add a migration that creates them."
        )


def test_upgrade_from_base_preserves_fk_ondelete_actions(
    isolated_db_settings,
) -> None:
    """Every foreign key's ON DELETE action must survive the migration chain.

    The app relies on the database to cascade deletes (the ORM relationships
    set `passive_deletes=True`), so a dropped `ON DELETE CASCADE` is the one
    way this design can lose or orphan data. `batch_alter_table` recreates a
    whole table to change one column, which is exactly where a clause can go
    missing; three tables in the delete chain have already been through it.
    """
    import app.models  # noqa: F401  # populates Base.metadata
    from app.db.base import Base

    upgrade_to_head()
    insp = inspect(_engine_for(isolated_db_settings))

    for table_name, table in Base.metadata.tables.items():
        expected = {
            (fk.parent.name, (fk.ondelete or "").upper())
            for fk in table.foreign_keys
        }
        live = {
            (col, (fk["options"].get("ondelete") or "").upper())
            for fk in insp.get_foreign_keys(table_name)
            for col in fk["constrained_columns"]
        }
        assert expected <= live, (
            f"Table {table_name!r} lost foreign key ON DELETE actions: "
            f"model expects {sorted(expected - live)}, live schema has "
            f"{sorted(live)}."
        )
