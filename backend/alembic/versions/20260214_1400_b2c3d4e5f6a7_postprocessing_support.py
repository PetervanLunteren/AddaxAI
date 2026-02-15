"""add postprocessing support

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-14 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop classification_all_probs from detections — raw predictions live in JSON files
    op.drop_column("detections", "classification_all_probs")

    # Add postprocessing_settings_hash to projects
    # SHA-256 hash of last-applied smoothing settings; NULL means never processed
    op.add_column(
        "projects",
        sa.Column("postprocessing_settings_hash", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("projects", "postprocessing_settings_hash")
    op.add_column(
        "detections",
        sa.Column("classification_all_probs", sa.JSON, nullable=True),
    )
