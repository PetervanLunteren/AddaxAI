"""deployment_queue use_file_mtime_fallback

Add a ``use_file_mtime_fallback`` boolean to ``deployment_queue``. Set by
the opt-in checkbox in the folder scan, which is only offered when the
scan found no capture dates at all and which shows the user the exact
date range file timestamps would produce before they tick it. The ingest
then fills in `File.captured_at_local` from each file's modification time,
but only for files whose metadata carries no date: a real capture date
always wins.

It lives on the queue entry rather than on ``deployments`` because the
user makes the choice at queue-add time and the worker reads it minutes
or days later, across app restarts, and the queue row is the only one
alive at both moments. Unlike ``datetime_offset_seconds`` it is not
mirrored onto the deployment: it is a one-shot ingest decision, not
re-editable after the fact, and it is deliberately not displayed.

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are each
skipped when the live schema is already in the target state.

Revision ID: 4d5e6f7a8b9c
Revises: 3c4d5e6f7a8b
Create Date: 2026-07-30 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4d5e6f7a8b9c"
down_revision: str | None = "3c4d5e6f7a8b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _queue_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("deployment_queue")}


def upgrade() -> None:
    bind = op.get_bind()
    if "use_file_mtime_fallback" not in _queue_columns(bind):
        op.add_column(
            "deployment_queue",
            sa.Column(
                "use_file_mtime_fallback",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "use_file_mtime_fallback" in _queue_columns(bind):
        op.execute(
            "ALTER TABLE deployment_queue DROP COLUMN use_file_mtime_fallback"
        )
