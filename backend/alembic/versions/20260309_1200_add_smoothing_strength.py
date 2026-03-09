"""Add smoothing_strength column to projects.

Stores the smoothing aggressiveness preset (mild, normal, aggressive).
Existing projects get "normal" which matches current MegaDetector defaults.

Revision ID: e5f6a7b8c9d1
Revises: d4e5f6a7b8c0
Create Date: 2026-03-09 12:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d1"
down_revision: Union[str, None] = "d4e5f6a7b8c0"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(
            sa.Column(
                "smoothing_strength",
                sa.String(20),
                nullable=False,
                server_default="normal",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("smoothing_strength")
