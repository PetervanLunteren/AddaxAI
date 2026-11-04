"""calibration_answers table

Answers to "Is this a <label>?" from the Calibration page: the AI's
label and score at the moment of answering, and right / wrong / unsure.
One row per detection, removed with it.

New table, no rows touched. Guarded against drifted beta DBs
(DEVELOPERS.md): create and drop are each skipped when the live schema
is already in the target state.

Revision ID: d4e5f6a7b8ca
Revises: c3d4e5f6a7b9
Create Date: 2026-11-03 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8ca"
down_revision: str | None = "c3d4e5f6a7b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "calibration_answers"


def upgrade() -> None:
    if _TABLE in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        _TABLE,
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("detection_id", sa.String(length=36), nullable=False),
        sa.Column("label", sa.String(length=100), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("answer", sa.String(length=10), nullable=False),
        sa.Column("answered_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["detection_id"], ["detections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_calibration_answers_detection", _TABLE, ["detection_id"], unique=True
    )
    op.create_index("idx_calibration_answers_label", _TABLE, ["label"])


def downgrade() -> None:
    if _TABLE in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table(_TABLE)
