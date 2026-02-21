"""add frame source fields to files

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-02-20 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b8c9d0e1f2a3"
down_revision: Union[str, None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("files") as batch_op:
        batch_op.add_column(
            sa.Column("source_video_id", sa.String(36), nullable=True)
        )
        batch_op.add_column(
            sa.Column("source_frame_number", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_files_source_video",
            "files",
            ["source_video_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index(
            "idx_files_source_video", ["source_video_id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("files") as batch_op:
        batch_op.drop_index("idx_files_source_video")
        batch_op.drop_constraint("fk_files_source_video", type_="foreignkey")
        batch_op.drop_column("source_frame_number")
        batch_op.drop_column("source_video_id")
