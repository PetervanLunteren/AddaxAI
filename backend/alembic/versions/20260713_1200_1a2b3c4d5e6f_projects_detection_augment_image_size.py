"""add detection_augment and detection_image_size to projects

Two advanced MegaDetector inference options, mirroring the existing
detection_batch_size plumbing:

- `detection_augment` (bool): run the detector on augmented copies and
  merge the results. Existing rows backfill to False.
- `detection_image_size` (int, nullable): override the long-edge resize
  size; NULL means MegaDetector's model-native default.

Idempotent: skips the ADD COLUMN steps if the column already exists, so a
half-applied attempt (column added, alembic_version not updated) does not
loop on startup with a duplicate-column error.

Revision ID: 1a2b3c4d5e6f
Revises: f8a9b0c1d2e3
Create Date: 2026-07-13 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1a2b3c4d5e6f"
down_revision: str | None = "f8a9b0c1d2e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _projects_has_column(name: str) -> bool:
    inspector = inspect(op.get_bind())
    return any(c["name"] == name for c in inspector.get_columns("projects"))


def upgrade() -> None:
    if not _projects_has_column("detection_augment"):
        # server_default backfills existing rows to False. Future rows take
        # their default from the SQLAlchemy model, so the server default is
        # only load-bearing during this migration.
        op.add_column(
            "projects",
            sa.Column(
                "detection_augment",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )
    if not _projects_has_column("detection_image_size"):
        op.add_column(
            "projects",
            sa.Column(
                "detection_image_size",
                sa.Integer(),
                nullable=True,
            ),
        )


def downgrade() -> None:
    if _projects_has_column("detection_image_size"):
        op.drop_column("projects", "detection_image_size")
    if _projects_has_column("detection_augment"):
        op.drop_column("projects", "detection_augment")
