"""drop projects.taxonomic_rollup_threshold

The taxonomic rollup confidence threshold is fixed policy
(app.core.confidence.ROLLUP_THRESHOLD = 0.65), not a per-project
setting: it was never rendered in the UI and every project carried the
same value. Drop the column.

Guarded against drifted beta DBs (DEVELOPERS.md): the drop is skipped
when the column is already gone, and the downgrade re-adds it with the
old default only when absent.

Revision ID: e7f8a9b0c1d2
Revises: d6e7f8a9b0c1
Create Date: 2026-07-08 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7f8a9b0c1d2"
down_revision: str | None = "d6e7f8a9b0c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _projects_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("projects")}


def upgrade() -> None:
    bind = op.get_bind()
    if "taxonomic_rollup_threshold" in _projects_columns(bind):
        op.execute(
            "ALTER TABLE projects DROP COLUMN taxonomic_rollup_threshold"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "taxonomic_rollup_threshold" not in _projects_columns(bind):
        op.execute(
            "ALTER TABLE projects "
            "ADD COLUMN taxonomic_rollup_threshold FLOAT NOT NULL "
            "DEFAULT 0.65"
        )
