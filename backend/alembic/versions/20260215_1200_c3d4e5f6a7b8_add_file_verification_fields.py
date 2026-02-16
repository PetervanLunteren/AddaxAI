"""add file verification fields

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-02-15 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("files") as batch_op:
        batch_op.add_column(
            sa.Column("verified", sa.Boolean(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("verified_at", sa.DateTime(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("notes", sa.Text(), nullable=True)
        )
        batch_op.create_index("idx_files_verified", ["verified"])


def downgrade() -> None:
    with op.batch_alter_table("files") as batch_op:
        batch_op.drop_index("idx_files_verified")
        batch_op.drop_column("notes")
        batch_op.drop_column("verified_at")
        batch_op.drop_column("verified")
