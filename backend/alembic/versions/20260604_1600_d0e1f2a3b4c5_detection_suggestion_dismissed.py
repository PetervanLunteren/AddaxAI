"""detection suggestion_dismissed

Add a ``suggestion_dismissed`` boolean to ``detections``. Set by the
"Dismiss" button on a suggestion cohort: it hides the cohort from the
suggestions review without touching labels or verified state. Filtered
in Python during cohort grouping, never in a SQL WHERE, so no index is
needed (mirrors ``verified``).

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are each
skipped when the live schema is already in the target state.

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-06-04 16:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d0e1f2a3b4c5"
down_revision: str | None = "c9d0e1f2a3b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _detections_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("detections")}


def upgrade() -> None:
    bind = op.get_bind()
    if "suggestion_dismissed" not in _detections_columns(bind):
        op.add_column(
            "detections",
            sa.Column(
                "suggestion_dismissed",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "suggestion_dismissed" in _detections_columns(bind):
        op.execute("ALTER TABLE detections DROP COLUMN suggestion_dismissed")
