"""Tests for `app.db.migrations` and the init_db flow.

`alembic_version` is ground truth. The app runs `alembic upgrade head`
and then checks the result against `Base.metadata`; it never guesses a
revision and never re-stamps backwards to replay the chain, because that
replay re-ran one-time data migrations over already-migrated data and
destroyed user verifications on 2026-05-27.

So the assertions here are: the chain from base produces exactly the
schema the models declare, `schema_problems` notices each way that can
fail, and every database the app refuses is left untouched.
"""


import pytest
from fastapi import FastAPI
from sqlalchemy import inspect, text

from app.core.startup_error import (
    GENERIC_STARTUP_FAILURE,
    STARTUP_ERROR_FILENAME,
    write_startup_error,
)
from app.db.base import init_db
from app.db.migrations import (
    SchemaError,
    ensure_upgradable,
    get_current_revision,
    get_head_revision,
    schema_problems,
    upgrade_to_head,
)


def _row_counts(engine) -> dict[str, int]:
    """Row count per user table, for asserting nothing was touched.

    One connection for the whole walk. Calling `engine.connect()` per
    table leaks a connection each time and exhausts the pool once these
    tests share an engine, which showed up only in a full-suite run.
    """
    insp = inspect(engine)
    with engine.connect() as conn:
        return {
            name: conn.execute(
                text(f"SELECT COUNT(*) FROM {name}")  # noqa: S608 - table names
            ).scalar()
            for name in insp.get_table_names()
        }


# ---------------------------------------------------------------------------
# The migration chain must produce the schema the models declare
# ---------------------------------------------------------------------------


def test_upgrade_from_base_matches_models(engine) -> None:
    """Running the whole chain from base must satisfy `schema_problems`.

    This is the immutability guard, and it is the specification the
    runtime check is built from: `init_db` refuses to start on exactly
    what this test asserts is empty. It catches a migration that drifts
    from `Base.metadata` (references a column the chain never creates,
    forgets an index, or loses a foreign key's ON DELETE action), which
    is the class of bug that produced issue #11.
    """
    upgrade_to_head()
    assert schema_problems(engine) == []


# Deliberately absent: a "the chain is idempotent" test. A second
# `upgrade head` re-runs nothing, because alembic no-ops at head, so such
# a test asserts nothing while reading as coverage. The property itself
# is unreachable anyway: stamping an at-head database back to base and
# replaying fails on the first migration with "table audit_log already
# exists", and the design does not depend on it. See DEVELOPERS.md.


# ---------------------------------------------------------------------------
# schema_problems: one case per way the live schema can fall short
# ---------------------------------------------------------------------------


def test_schema_problems_reports_a_missing_table(engine) -> None:
    upgrade_to_head()
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.execute(text("DROP TABLE audit_log"))

    assert "missing table audit_log" in schema_problems(engine)


def test_schema_problems_reports_a_missing_column(engine) -> None:
    upgrade_to_head()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE deployments DROP COLUMN warnings"))

    assert "missing column deployments.warnings" in schema_problems(engine)


def test_schema_problems_reports_a_missing_index(engine) -> None:
    """The foreign-key indexes nobody queries through are the point.

    `idx_event_obs_max_n_file` covers a foreign key's child column. Its
    absence is invisible to every query and shows up only as a delete
    that takes hours, so a plain table-and-column check would miss it.
    """
    upgrade_to_head()
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX idx_event_obs_max_n_file"))

    problems = schema_problems(engine)
    assert any("idx_event_obs_max_n_file" in p for p in problems), problems


def test_schema_problems_reports_a_lost_fk_ondelete(engine) -> None:
    """A dropped ON DELETE CASCADE is the one way this design loses rows.

    The ORM sets `passive_deletes=True` everywhere and lets the database
    do the cascade, so a foreign key that comes back from a
    `batch_alter_table` without its ON DELETE clause silently orphans
    children. Alembic's own `compare_metadata` does not report this,
    which is why `schema_problems` walks foreign keys itself.
    """
    upgrade_to_head()
    # Rebuild event_observations with the ON DELETE clause stripped off
    # its event_id foreign key, which is what a careless batch migration
    # produces.
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.execute(text("ALTER TABLE event_observations RENAME TO _eo_old"))
        conn.execute(
            text(
                "CREATE TABLE event_observations ("
                "id VARCHAR(36) NOT NULL PRIMARY KEY, "
                "event_id VARCHAR(36) NOT NULL REFERENCES events(id))"
            )
        )
        conn.execute(text("DROP TABLE _eo_old"))

    problems = schema_problems(engine)
    assert any(
        "event_observations.event_id" in p and "ON DELETE CASCADE" in p
        for p in problems
    ), problems


def test_schema_problems_reports_a_missing_unique_constraint(engine) -> None:
    """A lost unique constraint is the only failure that stays silent.

    A missing column or index announces itself (a crash, or a delete
    that takes hours). A missing unique constraint just lets duplicate
    rows accumulate, so it is worth catching at startup.
    """
    upgrade_to_head()
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.execute(text("ALTER TABLE event_observations RENAME TO _eo_old"))
        conn.execute(
            text(
                "CREATE TABLE event_observations ("
                "id VARCHAR(36) NOT NULL PRIMARY KEY, "
                "event_id VARCHAR(36) NOT NULL "
                "REFERENCES events(id) ON DELETE CASCADE)"
            )
        )
        conn.execute(text("DROP TABLE _eo_old"))

    problems = schema_problems(engine)
    assert any(
        "uq_event_obs_event_taxonomy" in p for p in problems
    ), problems


def test_schema_problems_reports_an_unreadable_schema(engine) -> None:
    """A schema too broken to reflect is a problem, not a crash.

    `compare_metadata` reflects the whole schema, so a foreign key
    pointing at a table that no longer exists makes it raise. Renaming
    a table carries other tables' foreign key clauses along with it, so
    dropping the renamed original leaves exactly that state. The user
    should get the normal refusal with the recovery options, not an
    unexplained stack trace.
    """
    upgrade_to_head()
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        # deployments.site_id follows the rename to _sites_old, then
        # points at nothing once that table is dropped.
        conn.execute(text("ALTER TABLE sites RENAME TO _sites_old"))
        conn.execute(text("DROP TABLE _sites_old"))

    problems = schema_problems(engine)
    assert any("could not be read" in p for p in problems), problems


def test_schema_problems_ignores_extra_live_columns(engine) -> None:
    """Only "the models have it, the database doesn't" is a problem.

    A column the live schema has and the models don't is what a
    half-applied DROP COLUMN looks like. It is harmless, and treating it
    as a failure would refuse a healthy user's launch.
    """
    upgrade_to_head()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE projects ADD COLUMN leftover TEXT"))

    assert schema_problems(engine) == []


# ---------------------------------------------------------------------------
# init_db: fresh install and the healthy steady state
# ---------------------------------------------------------------------------


def test_init_db_builds_a_fresh_install_from_base(engine) -> None:
    """An empty database reaches head with a schema matching the models."""
    init_db()

    assert get_current_revision(engine) == get_head_revision()
    assert schema_problems(engine) == []


def test_init_db_noop_on_already_healthy_db(engine) -> None:
    """A DB already at head must reach the second init_db unchanged."""
    init_db()
    head = get_head_revision()
    assert get_current_revision(engine) == head

    init_db()  # second init_db: must be a no-op
    assert get_current_revision(engine) == head


# ---------------------------------------------------------------------------
# init_db: every refusal must leave the database untouched
# ---------------------------------------------------------------------------


def test_init_db_refuses_a_db_with_no_alembic_version(engine) -> None:
    """A database from before the 2026-05-08 alembic wiring is refused.

    It used to be adopted by stamping it at a revision guessed from its
    schema. Guessing is what this design removes, and such a database
    cannot exist unless it predates alembic entirely, so it gets the
    same restore-or-start-fresh message as any other unusable file.
    """
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE files (id TEXT PRIMARY KEY, captured_at TEXT)")
        )

    with pytest.raises(SchemaError, match=r"early AddaxAI beta"):
        init_db()

    # Left untouched: no alembic_version row written, no migration run.
    assert get_current_revision(engine) is None
    assert set(inspect(engine).get_table_names()) == {"files"}


def test_init_db_refuses_an_unknown_stamped_revision(engine) -> None:
    """A revision that is not on disk means a different build wrote this.

    Alembic raises CommandError while resolving the chain, before any
    migration runs, so the database is untouched when we refuse.
    """
    init_db()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE alembic_version SET version_num = 'zzzz99999999'")
        )
    before = _row_counts(engine)

    with pytest.raises(SchemaError, match=r"different version of AddaxAI"):
        init_db()

    assert get_current_revision(engine) == "zzzz99999999"
    assert _row_counts(engine) == before


def test_init_db_refuses_an_ambiguous_alembic_version(engine) -> None:
    """More than one version row means the version table itself is broken.

    `get_current_revision` would silently read whichever row SQLite
    returned first, so the stamp we are about to trust absolutely would
    be a coin flip.
    """
    init_db()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES ('abc123')")
        )

    with pytest.raises(SchemaError, match=r"different version of AddaxAI"):
        init_db()


def test_init_db_refuses_a_lying_stamp_without_repairing_it(engine) -> None:
    """The Cara case: stamped at head, schema missing a column.

    Alembic trusts the stamp, so `upgrade head` is a silent no-op and
    the column stays missing. The old code detected this and "repaired"
    it by re-stamping backwards and replaying the chain, which re-ran
    destructive data migrations. Now it must refuse, name the column,
    and change nothing.
    """
    init_db()
    head = get_head_revision()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE deployments DROP COLUMN warnings"))
    before = _row_counts(engine)

    with pytest.raises(SchemaError, match=r"missing column deployments.warnings"):
        init_db()

    # Nothing repaired, nothing replayed: same stamp, same rows, and the
    # column is still missing rather than silently re-added.
    assert get_current_revision(engine) == head
    assert _row_counts(engine) == before
    live = {c["name"] for c in inspect(engine).get_columns("deployments")}
    assert "warnings" not in live


def test_ensure_upgradable_passes_a_fresh_install(engine) -> None:
    """No tables at all is a fresh install, not a broken database."""
    ensure_upgradable(engine)  # must not raise


# ---------------------------------------------------------------------------
# Telling the user why: the startup error file
# ---------------------------------------------------------------------------


def test_write_startup_error_records_the_message(isolated_db_settings) -> None:
    write_startup_error(isolated_db_settings, "something went wrong")

    path = isolated_db_settings.user_data_dir / STARTUP_ERROR_FILENAME
    assert path.read_text(encoding="utf-8") == "something went wrong"


async def test_lifespan_writes_the_refusal_for_the_error_page(
    isolated_db_settings, monkeypatch
) -> None:
    """The refusal must reach the file Electron shows on the error page.

    This is the whole point of refusing rather than self-healing: the
    backend exits before the API or the frontend exist, so without this
    wire the user only ever sees "the backend stopped while starting up
    (exit code N)" and has nothing to act on.
    """
    from app.main import lifespan

    monkeypatch.setattr("app.main.get_settings", lambda: isolated_db_settings)

    def _refuse() -> None:
        raise SchemaError("this database is unusable, restore a backup")

    monkeypatch.setattr("app.main.init_db", _refuse)

    with pytest.raises(SchemaError):
        async with lifespan(FastAPI()):
            pass

    path = isolated_db_settings.user_data_dir / STARTUP_ERROR_FILENAME
    assert path.read_text(encoding="utf-8") == (
        "this database is unusable, restore a backup"
    )


async def test_lifespan_writes_a_generic_message_for_other_failures(
    isolated_db_settings, monkeypatch
) -> None:
    """A non-schema crash still gets a page, without leaking a traceback."""
    from app.main import lifespan

    monkeypatch.setattr("app.main.get_settings", lambda: isolated_db_settings)

    def _explode() -> None:
        raise RuntimeError("disk is full")

    monkeypatch.setattr("app.main.init_db", _explode)

    with pytest.raises(RuntimeError):
        async with lifespan(FastAPI()):
            pass

    path = isolated_db_settings.user_data_dir / STARTUP_ERROR_FILENAME
    written = path.read_text(encoding="utf-8")
    assert written == GENERIC_STARTUP_FAILURE
    assert "disk is full" not in written
