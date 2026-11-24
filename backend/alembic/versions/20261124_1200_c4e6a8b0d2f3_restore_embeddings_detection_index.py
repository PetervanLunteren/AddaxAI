"""restore idx_detection_embeddings_detection for the sort worker's plan

b9c3d5e7f9a1 dropped this index as a duplicate of
``idx_detection_embeddings_detection_model`` (detection_id,
embedding_model_id). For lookups and the cascade it is one. But the
Labels sort worker (``ml/inference/similarity_script.py``) runs on the
model env's SQLite (3.54 on 2026-11-24), and without the single-column
index that planner answers the similarity load with a scan of every
``detection_embeddings`` row, vectors included, and a sort of all of
them, where it used to walk ``idx_files_captured_at_local`` and stop at
the cap.

Measured on 2026-11-24 on the Elliot seed (2.57M boxes, 764K embeddings),
the worker run alone: the load step took 3 to 4 s before b9c3d5e7f9a1
and 31 to 35 s after it, the whole similarity sort 35 to 44 s against
62 to 68 s, the same with the old and the current script.

A new migration rather than an edit of b9c3d5e7f9a1: development
databases have already run that one.

Revision ID: c4e6a8b0d2f3
Revises: b9c3d5e7f9a1
Create Date: 2026-11-24 12:00:00

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e6a8b0d2f3"
down_revision: str | None = "b9c3d5e7f9a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "idx_detection_embeddings_detection",
        "detection_embeddings",
        ["detection_id"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_detection_embeddings_detection",
        table_name="detection_embeddings",
        if_exists=True,
    )
