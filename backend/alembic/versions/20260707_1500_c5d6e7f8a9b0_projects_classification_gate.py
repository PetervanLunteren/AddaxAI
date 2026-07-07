"""projects classification_gate

Add ``classification_gate`` to ``projects``: the detection confidence
above which animal crops are classified and embedded. Part of the
threshold decoupling (Dan Morris's beta feedback): MegaDetector now
runs untresholded (0.005) and the expensive per-crop model passes are
gated explicitly instead of implicitly by the old 0.1 MD floor.

Existing projects get 0.1, exactly the behaviour they had when the MD
floor played this role, so nothing changes for them retroactively.

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are
each skipped when the live schema is already in the target state.

Revision ID: c5d6e7f8a9b0
Revises: b4c5d6e7f8a9
Create Date: 2026-07-07 15:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c5d6e7f8a9b0"
down_revision: str | None = "b4c5d6e7f8a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _projects_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("projects")}


def upgrade() -> None:
    bind = op.get_bind()
    if "classification_gate" not in _projects_columns(bind):
        op.add_column(
            "projects",
            sa.Column(
                "classification_gate",
                sa.Float(),
                nullable=False,
                server_default="0.1",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "classification_gate" in _projects_columns(bind):
        op.execute("ALTER TABLE projects DROP COLUMN classification_gate")
