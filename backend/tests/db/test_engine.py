"""
One engine per process.

A new engine per request meant a new SQLite connection per request, with
the pragmas set again and an empty page cache each time. These tests pin
that the engine is shared, that it follows a changed database URL (the
test setup relies on that), and that a pooled connection still carries
the connect-time settings.
"""

from pathlib import Path

from sqlalchemy import text

from app.core.config import Settings
from app.db.base import (
    get_engine,
    get_session_factory,
    init_db,
    refresh_query_statistics,
)


def test_the_engine_is_built_once(isolated_db_settings):
    assert get_engine() is get_engine()
    assert get_session_factory() is get_session_factory()
    assert get_session_factory().kw["bind"] is get_engine()


def test_a_changed_database_url_gives_a_new_engine(
    isolated_db_settings, tmp_path: Path, monkeypatch
):
    first = get_engine()
    with first.connect() as conn:
        conn.execute(text("SELECT 1"))
    assert first.pool.checkedin() == 1

    other = Settings(
        user_data_dir=tmp_path / "other",
        database_url=f"sqlite:///{tmp_path / 'other' / 'addaxai.db'}",
    )
    monkeypatch.setattr("app.db.base.get_settings", lambda: other)

    second = get_engine()
    assert second is not first
    assert str(second.url) == other.database_url
    assert get_session_factory().kw["bind"] is second
    # The old engine was disposed, so it holds no connection to its file.
    assert first.pool.checkedin() == 0


def test_a_pooled_session_carries_the_connect_settings(isolated_db_settings):
    with get_session_factory()() as db:
        assert db.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert db.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert db.execute(text("SELECT seeded_hash('a', 1)")).scalar() != 0


def test_refreshing_statistics_replaces_every_pooled_connection(
    isolated_db_settings,
):
    """SQLite reads the statistics once per connection and keeps them.

    On 3.45.1 (the packaged build) a pooled connection would plan with
    old statistics forever, and the one that ran the ANALYZE keeps the
    deleted `idx_files_source_video` row in memory, which brings back the
    full-scan cascade on a project delete. The dev SQLite plans the same
    either way, so this pins the mechanism rather than a plan.
    """
    init_db()
    factory = get_session_factory()

    def raw(db):
        return db.connection().connection.dbapi_connection

    with factory() as db:
        idle = raw(db)
    with factory() as db:
        ran_analyze = raw(db)
        refresh_query_statistics(db)
        db.commit()
    with factory() as db:
        later = raw(db)

    assert later is not idle
    assert later is not ran_analyze


def test_concurrent_sessions_are_not_capped(isolated_db_settings):
    """The default pool fails the 16th concurrent session after 30 s.

    Slow requests the user has navigated away from keep their session,
    so that cap is reachable in the app.
    """
    sessions = [get_session_factory()() for _ in range(20)]
    try:
        for db in sessions:
            assert db.execute(text("SELECT 1")).scalar() == 1
    finally:
        for db in sessions:
            db.close()
