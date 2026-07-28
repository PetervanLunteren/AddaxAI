"""detection bbox nullable

Drop NOT NULL from `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height` on
the `detections` table so we can record event-level observations
(species seen in a video clip without a frame-anchored ROI). The data
contract is that the four columns are all-set or all-null per row;
Pydantic and a CHECK at the API layer enforce this — SQLite's column
nullability alone cannot express the joint constraint.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-14 15:00:00

"""
from collections.abc import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: str | None = 'a1b2c3d4e5f6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("detections") as batch_op:
        batch_op.alter_column("bbox_x", existing_type=None, nullable=True)
        batch_op.alter_column("bbox_y", existing_type=None, nullable=True)
        batch_op.alter_column("bbox_width", existing_type=None, nullable=True)
        batch_op.alter_column("bbox_height", existing_type=None, nullable=True)


def downgrade() -> None:
    # Coerce any event-level rows back to a zero bbox before re-applying
    # NOT NULL so the column reshape doesn't reject existing data. The
    # zero bbox is meaningless; the original observation is lossy after
    # downgrade — there's no way to know which rows were event-level.
    op.execute(
        "UPDATE detections "
        "SET bbox_x = 0, bbox_y = 0, bbox_width = 0, bbox_height = 0 "
        "WHERE bbox_x IS NULL"
    )
    with op.batch_alter_table("detections") as batch_op:
        batch_op.alter_column("bbox_x", existing_type=None, nullable=False)
        batch_op.alter_column("bbox_y", existing_type=None, nullable=False)
        batch_op.alter_column("bbox_width", existing_type=None, nullable=False)
        batch_op.alter_column("bbox_height", existing_type=None, nullable=False)
