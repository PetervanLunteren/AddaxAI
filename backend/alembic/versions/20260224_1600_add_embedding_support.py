"""Add embedding support.

Add embedding_model_id to projects table and create detection_embeddings table.

Revision ID: a1b2c3d4e5f6
Revises: 94d6210f7c39
Create Date: 2026-02-24 16:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "94d6210f7c39"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    # Add embedding_model_id to projects (SQLite requires batch_alter_table)
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(
            sa.Column("embedding_model_id", sa.String(length=100), nullable=True)
        )

    # Create detection_embeddings table
    op.create_table(
        "detection_embeddings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("detection_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("embedding_model_id", sa.String(length=100), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("l2_norm", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["detection_id"], ["detections.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_detection_embeddings_detection",
        "detection_embeddings",
        ["detection_id"],
    )
    op.create_index(
        "idx_detection_embeddings_model",
        "detection_embeddings",
        ["embedding_model_id"],
    )
    op.create_index(
        "idx_detection_embeddings_detection_model",
        "detection_embeddings",
        ["detection_id", "embedding_model_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("detection_embeddings")

    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("embedding_model_id")
