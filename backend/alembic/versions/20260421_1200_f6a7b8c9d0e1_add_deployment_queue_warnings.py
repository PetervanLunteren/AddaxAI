"""add warnings column to deployment_queue

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-04-21 12:00:00.000000

Adds a `warnings` Text column to `deployment_queue` for non-fatal
ingest issues (e.g. files skipped because they had no extractable
capture timestamp). Mirrors the existing `error` column; newline-joined
paths so the frontend can split and truncate consistently.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "deployment_queue",
        sa.Column("warnings", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("deployment_queue", "warnings")
