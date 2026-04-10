"""add batch size columns

Revision ID: a1b2c3d4e5f6
Revises: 66d29e0d6395
Create Date: 2026-04-09 14:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "66d29e0d6395"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("detection_batch_size", sa.Integer(), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("classification_batch_size", sa.Integer(), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("embedding_batch_size", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("projects", "embedding_batch_size")
    op.drop_column("projects", "classification_batch_size")
    op.drop_column("projects", "detection_batch_size")
