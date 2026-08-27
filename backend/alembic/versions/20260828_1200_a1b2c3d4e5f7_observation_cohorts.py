"""observation cohorts: sex, life stage and behaviour per row, notes per event

Three nullable columns on ``event_observations`` (``sex``, ``life_stage``,
``behavior``), one on ``events`` (``notes``), and the unique constraint on
``(event_id, label_taxonomy_id)`` goes. A species in an event is no
longer one row: a person can split it into cohorts (4 adult males, 2
juveniles), each with its own count and demographics. NULL is "unknown",
so no row is touched and nothing is backfilled.

Dropping a unique constraint on SQLite is a table rebuild, so that part
runs in batch mode, which every other constraint change in this chain
already uses. Both halves are guarded against a drifted DB
(DEVELOPERS.md, "Guard DDL anyway in new migrations"): a column that is
already there or a constraint that is already gone is a no-op, not a
crash.

Downgrade puts the constraint back, which fails while cohort rows exist.
That is the honest outcome: the data cannot be expressed in the old
schema.

Revision ID: a1b2c3d4e5f7
Revises: 9c0d1e2f3a4b
Create Date: 2026-08-28 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f7"
down_revision: str | None = "9c0d1e2f3a4b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "event_observations"
_CONSTRAINT = "uq_event_obs_event_taxonomy"
_COLUMNS = (
    (_TABLE, sa.Column("sex", sa.String(20), nullable=True)),
    (_TABLE, sa.Column("life_stage", sa.String(20), nullable=True)),
    (_TABLE, sa.Column("behavior", sa.String(50), nullable=True)),
    ("events", sa.Column("notes", sa.Text(), nullable=True)),
)


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def _has_constraint(bind) -> bool:
    return any(
        c["name"] == _CONSTRAINT
        for c in sa.inspect(bind).get_unique_constraints(_TABLE)
    )


def upgrade() -> None:
    bind = op.get_bind()
    for table, column in _COLUMNS:
        if column.name not in _columns(bind, table):
            op.add_column(table, column)

    if _has_constraint(bind):
        with op.batch_alter_table(_TABLE) as batch:
            batch.drop_constraint(_CONSTRAINT, type_="unique")


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_constraint(bind):
        with op.batch_alter_table(_TABLE) as batch:
            batch.create_unique_constraint(
                _CONSTRAINT, ["event_id", "label_taxonomy_id"]
            )

    for table, column in reversed(_COLUMNS):
        if column.name in _columns(bind, table):
            op.execute(f"ALTER TABLE {table} DROP COLUMN {column.name}")
