"""add observation_type and classification_method

Revision ID: a1b2c3d4e5f6
Revises: 477eebb281a3
Create Date: 2026-02-14 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "477eebb281a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add observation_type to files (default "unclassified" for existing rows)
    op.add_column(
        "files",
        sa.Column(
            "observation_type",
            sa.String(20),
            nullable=False,
            server_default="unclassified",
        ),
    )
    op.create_index("idx_files_observation_type", "files", ["observation_type"])

    # Add classification_method to detections
    op.add_column(
        "detections",
        sa.Column("classification_method", sa.String(20), nullable=True),
    )

    # Backfill observation_type for existing files based on their detections.
    # Priority: animal > person(human) > vehicle
    #
    # Files with animal detections -> "animal"
    op.execute(
        """
        UPDATE files SET observation_type = 'animal'
        WHERE id IN (
            SELECT DISTINCT file_id FROM detections WHERE category = 'animal'
        )
        """
    )
    # Files with person detections (but no animal) -> "human"
    op.execute(
        """
        UPDATE files SET observation_type = 'human'
        WHERE observation_type = 'unclassified'
        AND id IN (
            SELECT DISTINCT file_id FROM detections WHERE category = 'person'
        )
        """
    )
    # Files with vehicle detections (but no animal/person) -> "vehicle"
    op.execute(
        """
        UPDATE files SET observation_type = 'vehicle'
        WHERE observation_type = 'unclassified'
        AND id IN (
            SELECT DISTINCT file_id FROM detections WHERE category = 'vehicle'
        )
        """
    )

    # Backfill classification_method for detections that already have species set
    op.execute(
        """
        UPDATE detections SET classification_method = 'machine'
        WHERE species IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_index("idx_files_observation_type", table_name="files")
    op.drop_column("files", "observation_type")
    op.drop_column("detections", "classification_method")
