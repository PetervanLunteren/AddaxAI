"""add media_filter to projects

Which media a new analysis reads off disk: "all" | "images" | "videos".
Inference-time, like video_fps and detection_augment. The motivating case is
a camera left in video mode by mistake: "images" skips the videos rather than
making the user move files around by hand.

Existing rows backfill to "all", which is today's behaviour, so the upgrade is
a no-op for every current project.

Idempotent: skips the ADD COLUMN step if the column already exists, so a
half-applied attempt (column added, alembic_version not updated) does not
loop on startup with a duplicate-column error.

Revision ID: 2b3c4d5e6f7a
Revises: 1a2b3c4d5e6f
Create Date: 2026-07-16 14:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2b3c4d5e6f7a"
down_revision: str | None = "1a2b3c4d5e6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _projects_has_column(name: str) -> bool:
    inspector = inspect(op.get_bind())
    return any(c["name"] == name for c in inspector.get_columns("projects"))


def upgrade() -> None:
    if not _projects_has_column("media_filter"):
        # server_default backfills existing rows to "all". Future rows take
        # their default from the SQLAlchemy model, so the server default is
        # only load-bearing during this migration.
        op.add_column(
            "projects",
            sa.Column(
                "media_filter",
                sa.String(length=16),
                nullable=False,
                server_default="all",
            ),
        )


def downgrade() -> None:
    if _projects_has_column("media_filter"):
        op.drop_column("projects", "media_filter")
