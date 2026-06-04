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
    """`projects` exists without the later columns → initial revision."""
    engine = _engine_for(isolated_db_settings)
    with engine.begin() as conn:
        # Only the columns from the initial migration: no
        # observations_max_detections, no warnings. We don't need the
        # full table here; the fingerprint only checks for the table /
        # column it knows about.
        conn.execute(text("CREATE TABLE projects (id TEXT PRIMARY KEY)"))

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
        # common_name (detections) was added by c9d0e1f2a3b4, the current
        # head and the newest detectable fingerprint. Drop it so the
        # schema truly looks initial and the fingerprint walk does not
        # stop at head.
        conn.execute(text("ALTER TABLE detections DROP COLUMN common_name"))

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
    """User tables present but no alembic_version → stamp at detected rev."""
    engine = _engine_for(isolated_db_settings)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE projects (id TEXT PRIMARY KEY)"))

    # No alembic_version yet.
    assert get_current_revision(engine) is None

    result = reconcile_alembic_version(engine)
    assert result == "9c173fff3bcd"
    assert get_current_revision(engine) == "9c173fff3bcd"
