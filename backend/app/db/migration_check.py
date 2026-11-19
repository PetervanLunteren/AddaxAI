"""Did a migration keep every row? The checks shared by the CI test
(`tests/db/test_migration_keeps_rows.py`, synthetic rows, no backups
needed) and the release-time script (`scripts/check_migration_on_backups.py`,
real backups when the machine has them).

`schema_problems()` proves the *layout* after an upgrade. These prove the
*rows* came through: a table rebuild that drops rows, or a constraint that
refuses real data, leaves the layout perfect and the user's work gone.

The comparison is deliberately coarse: row counts and primary keys per
table, plus SQLite's own integrity and foreign key checks. A data
migration is allowed to change values (that is its job, and
`test_migration_data.py` pins those per migration); it is never allowed
to lose or invent rows.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

# Tables that are bookkeeping, not user data.
_SKIP = frozenset({"alembic_version"})

# Migrations that change a table's row count on purpose, and which tables.
# Every other migration must keep every row and add none. Add an entry
# only together with a test in test_migration_data.py that pins what the
# change does. Read by the CI test and by the backups script, so the two
# cannot disagree.
ROWS_CHANGED_ON_PURPOSE: dict[str, frozenset[str]] = {
    # Drops detections without a box (the retired box-less observation
    # flow).
    "f2a3b4c5d6e7": frozenset({"detections"}),
    # Adds a builtin label row for each detector class that had boxes but
    # no row ("fish", "elasmobranch").
    "a8b2c4d6e8f0": frozenset({"label_taxonomy"}),
}


def rows_changed_on_purpose(revisions: Iterable[str]) -> frozenset[str]:
    """The tables whose row count the given migrations may change."""
    return frozenset().union(
        *(ROWS_CHANGED_ON_PURPOSE.get(rev, frozenset()) for rev in revisions)
    )


@dataclass(frozen=True)
class TableSnapshot:
    count: int
    keys: frozenset[tuple]


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_alembic_tmp_%' "
        "ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows if r[0] not in _SKIP]


def _primary_key(conn: sqlite3.Connection, table: str) -> list[str]:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    pk = sorted((c[5], c[1]) for c in cols if c[5] > 0)
    return [name for _pos, name in pk] or ["rowid"]


def snapshot(db_path: str) -> dict[str, TableSnapshot]:
    """Row count and the set of primary keys of every user table."""
    conn = sqlite3.connect(db_path)
    try:
        out: dict[str, TableSnapshot] = {}
        for table in _user_tables(conn):
            pk = ", ".join(_primary_key(conn, table))
            keys = frozenset(
                tuple(r) for r in conn.execute(f"SELECT {pk} FROM {table}")
            )
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            out[table] = TableSnapshot(count=count, keys=keys)
        return out
    finally:
        conn.close()


def sqlite_health(db_path: str) -> list[str]:
    """SQLite's own verdicts: integrity_check and foreign_key_check.
    Returns the problems, empty when the file is sound."""
    conn = sqlite3.connect(db_path)
    try:
        problems: list[str] = []
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            problems.append(f"integrity_check: {integrity}")
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            problems.append(f"{len(violations)} foreign key violation(s)")
        leftover = conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE '_alembic_tmp_%'"
        ).fetchall()
        if leftover:
            problems.append(f"batch-mode temp table left behind: {leftover[0][0]}")
        return problems
    finally:
        conn.close()


def lost_rows(
    before: dict[str, TableSnapshot],
    after: dict[str, TableSnapshot],
    changed_on_purpose: frozenset[str] = frozenset(),
) -> list[str]:
    """Every way a table came out different in rows: gone, fewer or more
    rows, or a primary key that vanished. A table a migration dropped on
    purpose is reported too; that is rare enough to want a human to look.
    Tables in ``changed_on_purpose`` (see ``rows_changed_on_purpose``) may
    gain or lose rows."""
    problems: list[str] = []
    for table, b in before.items():
        a = after.get(table)
        if a is None:
            problems.append(f"table {table} is gone")
            continue
        if table in changed_on_purpose:
            continue
        if a.count != b.count:
            problems.append(f"{table}: {b.count} rows before, {a.count} after")
        missing = b.keys - a.keys
        if missing:
            problems.append(f"{table}: {len(missing)} row(s) lost, e.g. {next(iter(missing))}")
    return problems
