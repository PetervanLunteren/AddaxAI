"""add the independent events cache

Two tables the statistics read independent events from (see
`app/models/independent_event_cache.py`). Every camera that has sequences
is marked stale, so each project's cache is filled by its first statistics
request after the update, not here: a full fill of a 900K-photo project
takes about 17 s, and the update must not make every first start slow.

The triggers that keep the cache fresh are not in this migration. They
are created by `init_db` at every start (`app/db/triggers.py`), because a
migration that rebuilds a table drops its triggers without a word, and a
shipped migration cannot be corrected later.

Idempotent: tables are created only when missing, and marking a camera
stale twice is one row.

Revision ID: e0a2c4e6f8b2
Revises: d9f1b3c5e7a1
Create Date: 2026-11-27 10:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e0a2c4e6f8b2"
down_revision: str | None = "d9f1b3c5e7a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    tables = set(inspect(op.get_bind()).get_table_names())
    if "independent_event_cache" not in tables:
        op.create_table(
            "independent_event_cache",
            sa.Column("observation_rowid", sa.Integer(), autoincrement=False, nullable=False),
            sa.Column("event_id", sa.Integer(), nullable=False),
            sa.Column("event_start_local", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("observation_rowid"),
        )
    if "independent_event_stale" not in tables:
        op.create_table(
            "independent_event_stale",
            sa.Column("deployment_id", sa.String(length=36), nullable=False),
            sa.Column("camera_key", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("deployment_id", "camera_key"),
        )
    op.execute(
        "INSERT OR IGNORE INTO independent_event_stale (deployment_id, camera_key) "
        "SELECT DISTINCT deployment_id, camera_key FROM sequences"
    )


def downgrade() -> None:
    tables = set(inspect(op.get_bind()).get_table_names())
    for table in ("independent_event_stale", "independent_event_cache"):
        if table in tables:
            op.drop_table(table)
