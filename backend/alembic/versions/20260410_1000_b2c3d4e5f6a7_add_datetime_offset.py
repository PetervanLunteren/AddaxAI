"""add datetime offset

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-10 10:00:00.000000

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
    op.add_column(
        "deployment_queue",
        sa.Column("datetime_offset_seconds", sa.Integer(), nullable=True),
    )
    op.add_column(
        "deployments",
        sa.Column("datetime_offset_seconds", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("deployments", "datetime_offset_seconds")
    op.drop_column("deployment_queue", "datetime_offset_seconds")
