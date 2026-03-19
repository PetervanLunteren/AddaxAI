"""Add thumbnail_path column to projects.

Stores the absolute path to the project card thumbnail image.
Populated either by user upload or auto-selected from best detection.

Revision ID: g7h8i9j0k1l2
Revises: a1b2c3d4e5f6
Create Date: 2026-03-19 12:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "g7h8i9j0k1l2"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(
            sa.Column("thumbnail_path", sa.Text, nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("thumbnail_path")
