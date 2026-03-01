"""Add detection-level verification fields.

Add verified (bool) and verified_at (datetime) to detections table.

Revision ID: b2c3d4e5f6a8
Revises: a1b2c3d4e5f6
Create Date: 2026-02-27 12:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a8"
down_revision: Union[str, None] = "a1b2c3d4e5f600"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    with op.batch_alter_table("detections") as batch_op:
        batch_op.add_column(
            sa.Column("verified", sa.Boolean(), nullable=False, server_default=sa.text("0"))
        )
        batch_op.add_column(
            sa.Column("verified_at", sa.DateTime(), nullable=True)
        )
        batch_op.create_index("idx_detections_verified", ["verified"])


def downgrade() -> None:
    with op.batch_alter_table("detections") as batch_op:
        batch_op.drop_index("idx_detections_verified")
        batch_op.drop_column("verified_at")
        batch_op.drop_column("verified")
