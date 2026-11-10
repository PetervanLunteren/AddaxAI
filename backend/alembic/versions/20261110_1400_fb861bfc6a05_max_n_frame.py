"""max_n_frame_number on event_observations

The frame the AI's MaxN was counted on, beside the file it already
stores. The Counts page jumps a video to it and the counts export
carries it with the wall-clock time it stands for (derived, never
stored: file time plus frame over frame rate). NULL for images, for
human-only rows and for every row written before this column; the next
MaxN rebuild fills it.

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are each
skipped when the live schema is already in the target state.

Revision ID: fb861bfc6a05
Revises: d660da8ab8fc
Create Date: 2026-11-10 14:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fb861bfc6a05"
down_revision: str | None = "d660da8ab8fc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    if "max_n_frame_number" not in _columns(bind, "event_observations"):
        op.add_column(
            "event_observations",
            sa.Column("max_n_frame_number", sa.Integer(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "max_n_frame_number" in _columns(bind, "event_observations"):
        op.execute("ALTER TABLE event_observations DROP COLUMN max_n_frame_number")
