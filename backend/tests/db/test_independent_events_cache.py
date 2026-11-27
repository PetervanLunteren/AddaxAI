"""The independent events cache on a real database file: startup keeps its
triggers and forgets it after a migration, and a locked database answers
live (app/db/base.init_db, app/api/crud/independent_events.refresh)."""

import sqlite3
import time

from sqlalchemy import text

from app.db.base import get_session_factory, init_db
from app.db.triggers import TRIGGERS
from tests.db.conftest import insert_row, seed_deployment, upgrade_to


def _triggers(engine) -> set[str]:
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ))}


def _count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()


def _seed_sequence(engine) -> str:
    with engine.begin() as conn:
        project_id, deployment_id = seed_deployment(conn)
        sequence_id = insert_row(
            conn, "sequences", deployment_id=deployment_id, camera_key="/d/a/",
            sequence_start_local="2024-06-01 12:00:00",
            sequence_end_local="2024-06-01 12:00:00",
        )
        insert_row(conn, "sequence_observations", sequence_id=sequence_id,
                   label="deer", category="animal", max_n=2)
    return project_id


def test_startup_keeps_the_cache_unless_a_migration_ran(engine) -> None:
    """A normal start creates the triggers and keeps a filled cache. A start
    that migrated empties it and marks every camera stale: a migration may
    rebuild a table, which renumbers the rowids the cache is keyed on."""
    from app.api.crud.independent_events import refresh

    upgrade_to("d9f1b3c5e7a1")
    project_id = _seed_sequence(engine)

    init_db()  # migrates to head
    assert _triggers(engine) == set(TRIGGERS)
    assert _count(engine, "independent_event_stale") == 1
    db = get_session_factory()()
    refresh(db, project_id)
    db.close()
    assert _count(engine, "independent_event_cache") == 1
    assert _count(engine, "independent_event_stale") == 0

    init_db()  # nothing to migrate
    assert _triggers(engine) == set(TRIGGERS)
    assert _count(engine, "independent_event_cache") == 1

    # Pretend the last migration is still pending: the next start runs it
    # again (it is idempotent) and must forget the cache.
    with engine.begin() as conn:
        conn.exec_driver_sql("UPDATE alembic_version SET version_num = 'd9f1b3c5e7a1'")
    init_db()
    assert _count(engine, "independent_event_cache") == 0
    assert _count(engine, "independent_event_stale") == 1


def test_a_start_that_dies_after_the_migration_leaves_no_trusted_cache(
    engine, monkeypatch
) -> None:
    """The upgrade commits on its own, so a start can die right after it.
    The next start finds nothing to migrate; it must still not read the old
    cache rows, whose rowids the migration may have given to other rows."""
    import app.db.migrations as migrations
    from app.api.crud.independent_events import refresh

    init_db()
    project_id = _seed_sequence(engine)
    db = get_session_factory()()
    refresh(db, project_id)
    db.close()
    assert _count(engine, "independent_event_cache") == 1

    with engine.begin() as conn:
        conn.exec_driver_sql("UPDATE alembic_version SET version_num = 'd9f1b3c5e7a1'")

    def die(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(migrations, "schema_problems", die)
        try:
            init_db()
        except KeyboardInterrupt:
            pass
    init_db()  # the next launch: nothing to migrate
    assert _count(engine, "independent_event_cache") == 0
    assert _count(engine, "independent_event_stale") == 1

    # Whatever the migration marked itself: an empty cache is never taken
    # for a fresh one.
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM independent_event_stale")
    init_db()
    assert _count(engine, "independent_event_stale") == 1


def test_a_migration_runs_without_the_triggers(engine, monkeypatch) -> None:
    """They are dropped before the upgrade, so a migration that rebuilds a
    table never trips over them, and created again after it."""
    import app.db.migrations as migrations

    upgrade_to("d9f1b3c5e7a1")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TRIGGER ie_observation_insert AFTER INSERT ON "
                             "sequence_observations BEGIN SELECT 1; END")
    real_upgrade = migrations.upgrade_to_head
    seen_during_upgrade = []

    def spying_upgrade(*args, **kwargs):
        seen_during_upgrade.append(_triggers(engine))
        return real_upgrade(*args, **kwargs)

    monkeypatch.setattr(migrations, "upgrade_to_head", spying_upgrade)
    init_db()
    assert seen_during_upgrade == [set()]
    assert _triggers(engine) == set(TRIGGERS)


def test_a_locked_database_answers_live(engine) -> None:
    """A job holding SQLite's write lock does not make a statistic wait or
    fail: the refresh gives up after a quarter second and the events are
    calculated live, the camera staying stale for the next request."""
    from app.api.crud.independent_events import independent_events
    from app.models import independent_event_cache

    init_db()
    project_id = _seed_sequence(engine)

    blocker = sqlite3.connect(engine.url.database, timeout=0)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        db = get_session_factory()()
        started = time.monotonic()
        ie = independent_events(db, project_id)
        assert time.monotonic() - started < 2  # not the 30 s busy timeout
        assert ie is not independent_event_cache
        assert len(db.execute(text("SELECT 1")).all()) == 1
        rows = db.execute(ie.select()).all()
        db.close()
    finally:
        blocker.rollback()
        blocker.close()
    assert len(rows) == 1
    assert _count(engine, "independent_event_stale") == 1


def test_nothing_vacuums_the_main_database() -> None:
    """VACUUM may renumber rowids, and the cache is keyed on them. Only the
    package file is vacuumed (a separate database). Vacuuming the main
    database needs the cache emptied and every camera marked stale first
    (`app/db/triggers.py`); this test is the reminder."""
    from pathlib import Path

    app_dir = Path(__file__).resolve().parents[2] / "app"
    allowed = {app_dir / "services" / "project_package.py"}
    vacuuming = [
        str(path.relative_to(app_dir))
        for path in app_dir.rglob("*.py")
        if path not in allowed and "VACUUM" in path.read_text()
    ]
    assert vacuuming == []
