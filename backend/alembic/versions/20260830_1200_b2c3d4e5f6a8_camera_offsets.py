"""camera_offsets on deployments and deployment_queue

A per-camera clock correction for paired cameras: ``{"<subfolder>": seconds}``
added on top of ``datetime_offset_seconds`` for the files in that subfolder.
Two dependent cameras rarely agree on the time, and the cross-camera event
grouping only lines up once each camera is corrected on its own.

On both tables like the base offset: chosen at queue-add time, read by the
worker at ingest, editable on the deployment later. ``server_default`` is
needed because the tables are populated; SQLite stores ``{}`` as text and
the JSON type reads it back as an empty dict.

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are each
skipped when the live schema is already in the target state.

Revision ID: b2c3d4e5f6a8
Revises: a1b2c3d4e5f7
Create Date: 2026-08-30 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a8"
down_revision: str | None = "a1b2c3d4e5f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("deployments", "deployment_queue")


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    for table in _TABLES:
        if "camera_offsets" not in _columns(bind, table):
            op.add_column(
                table,
                sa.Column(
                    "camera_offsets",
                    sa.JSON(),
                    nullable=False,
                    server_default="{}",
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    for table in _TABLES:
        if "camera_offsets" in _columns(bind, table):
            op.execute(f"ALTER TABLE {table} DROP COLUMN camera_offsets")
