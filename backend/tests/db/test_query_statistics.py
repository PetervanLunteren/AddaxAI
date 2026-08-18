"""Tests for `refresh_query_statistics`.

`ANALYZE` writes one row into `sqlite_stat1` per index, whether anything
uses it or not. One of those rows must not survive: the one for
`idx_files_source_video`, the index behind the schema's only
self-referencing foreign key. See the comment on `_UNUSABLE_STAT_INDEX`
in `app/db/base.py` for what it costs.

**These assertions are about the row, not about the query plan.** A test
that asserted the plan would pass on the development and CI SQLite
whatever the code did: 3.53 uses the index either way, and only 3.45,
which the packaged build carries, answers `SCAN files` when the row is
present. So a plan assertion here would be green while the shipped app
was broken, which is worse than no test. The row is the thing that is
true on every version.
"""

import re
from pathlib import Path

from sqlalchemy import text

from app.db.base import _UNUSABLE_STAT_INDEX, init_db, refresh_query_statistics

from .conftest import insert_row, seed_deployment


def _stat_indexes(conn) -> set[str]:
    """Every index `sqlite_stat1` currently holds a statistic for."""
    rows = conn.execute(text("SELECT idx FROM sqlite_stat1")).all()
    return {r[0] for r in rows if r[0] is not None}


def _seed_files(conn, count: int) -> None:
    """`count` files under one deployment, none of them a video frame."""
    _, deployment_id = seed_deployment(conn)
    for i in range(count):
        insert_row(
            conn,
            "files",
            deployment_id=deployment_id,
            file_path=f"/tmp/img_{i:04d}.jpg",
            file_type="image",
            source_video_id=None,
        )


def test_refresh_drops_the_statistic_for_the_self_referencing_fk(engine):
    """The one row that turns a project delete into hours of scanning."""
    init_db()

    with engine.connect() as conn:
        _seed_files(conn, 40)
        conn.commit()

        conn.execute(text("ANALYZE"))
        conn.commit()
        assert _UNUSABLE_STAT_INDEX in _stat_indexes(conn), (
            "bare ANALYZE should write the row this test is about; if it no "
            "longer does, the test is passing vacuously"
        )

        refresh_query_statistics(conn)
        conn.commit()
        assert _UNUSABLE_STAT_INDEX not in _stat_indexes(conn)


def test_refresh_keeps_every_other_statistic(engine):
    """Dropping one row must not mean throwing the planner's stats away."""
    init_db()

    with engine.connect() as conn:
        _seed_files(conn, 40)
        conn.commit()

        refresh_query_statistics(conn)
        conn.commit()

        kept = _stat_indexes(conn)
        assert "idx_files_deployment" in kept
        assert _UNUSABLE_STAT_INDEX not in kept


def test_init_db_leaves_no_statistic_for_the_self_referencing_fk(engine):
    """The startup path, end to end: every launch re-analyses."""
    init_db()

    with engine.connect() as conn:
        _seed_files(conn, 40)
        conn.commit()
        conn.execute(text("ANALYZE"))
        conn.commit()

    # A second init_db is what a relaunch does.
    init_db()

    with engine.connect() as conn:
        assert _UNUSABLE_STAT_INDEX not in _stat_indexes(conn)


def test_a_fresh_connection_does_not_put_the_statistic_back(
    isolated_db_settings,
):
    """The pragmas run on every new connection and must not undo this.

    There used to be a `PRAGMA optimize` among them. It analyses only
    tables the connection has already queried and runs before the first
    statement, so on SQLite 3.45 it did nothing at all; on 3.50 and later
    it ran and rewrote this row, undoing the deletion on the very next
    connection. The pooled engine in the other tests hides that, because
    a reused connection never re-fires the `connect` event, so this one
    builds a brand new engine on purpose.
    """
    from sqlalchemy import create_engine

    init_db()
    fresh = create_engine(isolated_db_settings.database_url, future=True)
    try:
        with fresh.connect() as conn:
            _seed_files(conn, 40)
            conn.commit()
            refresh_query_statistics(conn)
            conn.commit()
    finally:
        fresh.dispose()

    # A second brand new engine: its connect-time pragmas are what would
    # write the row back.
    checker = create_engine(isolated_db_settings.database_url, future=True)
    try:
        with checker.connect() as conn:
            assert _UNUSABLE_STAT_INDEX not in _stat_indexes(conn)
    finally:
        checker.dispose()


def test_nothing_runs_a_bare_analyze(engine):
    """`ANALYZE` outside the helper puts the row straight back.

    This is the guard that matters over time. Deleting the row once is
    only correct for as long as every caller goes through
    `refresh_query_statistics`, and a new `db.execute(text("ANALYZE"))`
    somewhere would reintroduce the bug silently, with no failing test
    and no wrong output, just a delete that never finishes.
    """
    app_dir = Path(__file__).resolve().parents[2] / "app"
    pattern = re.compile(r"\bANALYZE\b")

    offenders = []
    for path in sorted(app_dir.rglob("*.py")):
        if path.name == "base.py" and path.parent.name == "db":
            continue  # the helper itself
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                rel = path.relative_to(app_dir.parent)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "call refresh_query_statistics() instead of ANALYZE:\n"
        + "\n".join(offenders)
    )
