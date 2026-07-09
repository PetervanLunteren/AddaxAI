"""rename projects.detection_threshold -> counting_threshold

The column is the in-app counting / visualization floor (what gets
counted and shown; verified detections always pass). "detection
threshold" was ambiguous — it read like MegaDetector's own floor
(app.core.confidence.MD_OUTPUT_CONFIDENCE_THRESHOLD = 0.005) rather
than the per-project counting knob. Rename it to match its global
default DEFAULT_COUNTING_THRESHOLD and its sibling classification_gate,
so every threshold is named for its purpose.

Guarded against drifted beta DBs (DEVELOPERS.md): the rename is skipped
when the new name is already present or the old name is gone; the
downgrade reverses it under the same guard.

Revision ID: f8a9b0c1d2e3
Revises: e7f8a9b0c1d2
Create Date: 2026-07-09 10:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f8a9b0c1d2e3"
down_revision: str | None = "e7f8a9b0c1d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _projects_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("projects")}


def upgrade() -> None:
    bind = op.get_bind()
    cols = _projects_columns(bind)
    if "detection_threshold" in cols and "counting_threshold" not in cols:
        op.execute(
            "ALTER TABLE projects "
            "RENAME COLUMN detection_threshold TO counting_threshold"
        )


def downgrade() -> None:
    bind = op.get_bind()
    cols = _projects_columns(bind)
    if "counting_threshold" in cols and "detection_threshold" not in cols:
        op.execute(
            "ALTER TABLE projects "
            "RENAME COLUMN counting_threshold TO detection_threshold"
        )
