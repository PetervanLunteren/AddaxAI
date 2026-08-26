"""paired_cameras on deployments and deployment_queue

Add a ``paired_cameras`` boolean to ``deployments`` and ``deployment_queue``.
Two or three cameras at one station watch the same spot; the user puts each
camera's files in a subfolder of one parent folder and adds the parent as one
deployment. With the flag on, event clustering and trap-night effort treat
the subfolders as one camera instead of separate cameras.

It lives on both tables like ``datetime_offset_seconds``: chosen at
queue-add time, read by the worker, and editable on the deployment later.

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are each
skipped when the live schema is already in the target state.

Revision ID: 9c0d1e2f3a4b
Revises: 8b9c0d1e2f3a
Create Date: 2026-08-25 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c0d1e2f3a4b"
down_revision: str | None = "8b9c0d1e2f3a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("deployments", "deployment_queue")


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    for table in _TABLES:
        if "paired_cameras" not in _columns(bind, table):
            op.add_column(
                table,
                sa.Column(
                    "paired_cameras",
                    sa.Boolean(),
                    nullable=False,
                    server_default="0",
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    for table in _TABLES:
        if "paired_cameras" in _columns(bind, table):
            op.execute(f"ALTER TABLE {table} DROP COLUMN paired_cameras")
