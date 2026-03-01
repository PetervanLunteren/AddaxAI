"""Add clustering settings to projects.

Add min_cluster_size and min_samples HDBSCAN params to projects table.

Revision ID: c3d4e5f6a7b9
Revises: b2c3d4e5f6a8
Create Date: 2026-02-27 14:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b9"
down_revision: Union[str, None] = "b2c3d4e5f6a8"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(
            sa.Column("min_cluster_size", sa.Integer(), nullable=False, server_default=sa.text("5"))
        )
        batch_op.add_column(
            sa.Column("min_samples", sa.Integer(), nullable=False, server_default=sa.text("3"))
        )


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("min_samples")
        batch_op.drop_column("min_cluster_size")
