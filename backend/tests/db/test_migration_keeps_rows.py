"""Every migration, run on a database that holds a row in every table.

`test_migrations.py` runs the chain on an *empty* database, so a table
rebuild that loses rows, or a new constraint that refuses the rows people
actually have, passes it. `test_migration_data.py` covers the migrations
someone wrote a test for. This closes the gap generically: for each
migration, stand the database up at the revision before it, put one row
in every table (foreign keys wired to each other), run that one step, and
check that no table lost a row and SQLite still calls the file sound.

Synthetic rows are simple rows, so this catches structural loss, not a
migration that mishandles one odd real value. That is what the
per-migration data tests are for, and what
`scripts/check_migration_on_backups.py` adds on a machine that has
backups. It needs no backups, so it runs everywhere, on every push.
"""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import text

from app.db.migration_check import lost_rows, snapshot, sqlite_health
from app.db.migrations import _alembic_config, get_current_revision
from tests.db.conftest import insert_row, upgrade_to

# Migrations that delete rows on purpose, and from which tables. Every
# other migration must keep every row. Add an entry here only together
# with a test in test_migration_data.py that pins what the deletion does.
_DELETES_ON_PURPOSE: dict[str, frozenset[str]] = {
    # Drops detections without a box (the retired box-less observation
    # flow).
    "f2a3b4c5d6e7": frozenset({"detections"}),
}

# Migrations run with foreign keys off (alembic/env.py, see DEVELOPERS.md
# "Database migrations"), so a DELETE in a migration does not cascade.
# f2a3b4c5d6e7 shipped relying on the cascade and is immutable: it leaves
# the embeddings of the detections it deletes as orphans, which every
# join ignores. Known, tolerated, and the reason new data migrations
# delete their children explicitly.
_ORPHANS_KNOWN = frozenset({"f2a3b4c5d6e7"})

# SQLite 3.45.x (the CI runner, the frozen macOS and Windows builds)
# reports a false `NULL value in <table>.<column>` from integrity_check
# for a REAL column added with `ADD COLUMN ... NOT NULL DEFAULT`, until
# the row is rewritten. The value reads back fine on every version and
# 3.46 fixed the verdict; see DEVELOPERS.md "Database migrations". The
# test reads the column instead, so a real NULL still fails.
_INTEGRITY_FALSE_POSITIVE: dict[str, tuple[str, str]] = {
    "c5d6e7f8a9b0": ("projects", "classification_gate"),
}


def _steps() -> list[tuple[str, str]]:
    """(from, to) for every migration in the chain, oldest first."""
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_alembic_config())
    steps = [
        (rev.down_revision, rev.revision)
        for rev in script.walk_revisions()
        if isinstance(rev.down_revision, str)
    ]
    return list(reversed(steps))


def _seed_every_table(db_path: str) -> None:
    """One row per user table, parents before children, foreign keys
    pointing at the row just made in the parent table. Discovered from
    the live schema, so it needs no per-revision table list."""
    conn = sqlite3.connect(db_path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
            )
        ]
        fks = {
            t: [
                (row[3], row[2])  # (from column, parent table)
                for row in conn.execute(f"PRAGMA foreign_key_list({t})")
            ]
            for t in tables
        }
    finally:
        conn.close()

    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    ids: dict[str, str] = {}
    pending = list(tables)
    with engine.begin() as sa_conn:
        sa_conn.execute(text("PRAGMA foreign_keys=ON"))
        while pending:
            progressed = False
            for table in list(pending):
                parents = {p for _c, p in fks[table] if p != table}
                if not parents <= set(ids):
                    continue
                values = {
                    col: ids[parent]
                    for col, parent in fks[table]
                    if parent != table
                }
                ids[table] = insert_row(sa_conn, table, **values)
                pending.remove(table)
                progressed = True
            assert progressed, f"circular foreign keys among {pending}"
    engine.dispose()


@pytest.mark.parametrize("step", _steps(), ids=lambda s: s[1])
def test_migration_keeps_every_row(engine, isolated_db_settings, step):
    before_rev, rev = step
    db_path = str(isolated_db_settings.user_data_dir / "addaxai.db")

    upgrade_to(before_rev)
    _seed_every_table(db_path)
    before = snapshot(db_path)
    assert all(t.count == 1 for t in before.values()), {
        k: v.count for k, v in before.items() if v.count != 1
    }

    upgrade_to(rev)

    assert get_current_revision(engine) == rev
    health = sqlite_health(db_path)
    if rev in _ORPHANS_KNOWN:
        health = [h for h in health if "foreign key" not in h]
    if rev in _INTEGRITY_FALSE_POSITIVE:
        table, column = _INTEGRITY_FALSE_POSITIVE[rev]
        health = [h for h in health if h != f"integrity_check: NULL value in {table}.{column}"]
        with sqlite3.connect(db_path) as conn:
            # typeof, not IS NULL: the planner folds IS NULL on a NOT
            # NULL column to false without reading a row.
            really_null = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE typeof({column}) = 'null'"
            ).fetchone()[0]
        assert really_null == 0
    assert health == []
    allowed = _DELETES_ON_PURPOSE.get(rev, frozenset())
    unexpected = [
        p for p in lost_rows(before, snapshot(db_path))
        if p.removeprefix("table ").split(":")[0].split(" ")[0] not in allowed
    ]
    assert unexpected == []
