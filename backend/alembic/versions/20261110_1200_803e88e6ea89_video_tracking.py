"""video_tracking on projects

Add a ``video_tracking`` boolean to ``projects``: follow each animal
through a video with a tracker so the Labels page reviews one card per
tracked animal instead of one per frame. Inference-time like
``video_fps``: read by a new analysis only, never by postprocessing.
Off by default; the form switches it on when an underwater detector is
chosen.

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are each
skipped when the live schema is already in the target state.

Revision ID: 803e88e6ea89
Revises: 674c256c8f32
Create Date: 2026-11-10 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "803e88e6ea89"
down_revision: str | None = "674c256c8f32"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    if "video_tracking" not in _columns(bind, "projects"):
        op.add_column(
            "projects",
            sa.Column(
                "video_tracking",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "video_tracking" in _columns(bind, "projects"):
        op.execute("ALTER TABLE projects DROP COLUMN video_tracking")
