"""add original_label and original_label_confidence to detections

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-04-22 10:00:00.000000

Preserves the raw top-1 classifier output per detection so user
relabels and postprocessing rollups do not destroy the original
prediction. Needed for the confusion matrix and classification
report insight views.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "detections",
        sa.Column("original_label", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "detections",
        sa.Column("original_label_confidence", sa.Float(), nullable=True),
    )
    op.create_index(
        "idx_detections_original_label",
        "detections",
        ["original_label"],
    )


def downgrade() -> None:
    op.drop_index("idx_detections_original_label", table_name="detections")
    op.drop_column("detections", "original_label_confidence")
    op.drop_column("detections", "original_label")
