"""deployments classification_gate_used

Add ``classification_gate_used`` to ``deployments``: the classification
gate the deployment was analysed with (the project's value at run
time). The project-level setting can change between analyses, so
mixed-gate projects need the per-run record to explain what was
classified / embedded. Audit metadata only, never queried for logic.
NULL marks deployments analysed before the gate existed.

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are
each skipped when the live schema is already in the target state.

Revision ID: d6e7f8a9b0c1
Revises: c5d6e7f8a9b0
Create Date: 2026-07-08 10:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d6e7f8a9b0c1"
down_revision: str | None = "c5d6e7f8a9b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _deployments_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("deployments")}


def upgrade() -> None:
    bind = op.get_bind()
    if "classification_gate_used" not in _deployments_columns(bind):
        op.add_column(
            "deployments",
            sa.Column("classification_gate_used", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "classification_gate_used" in _deployments_columns(bind):
        op.execute(
            "ALTER TABLE deployments DROP COLUMN classification_gate_used"
        )
