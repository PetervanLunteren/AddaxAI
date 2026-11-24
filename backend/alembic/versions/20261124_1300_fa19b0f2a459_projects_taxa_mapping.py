"""add taxa_mapping to projects

A custom SpeciesNet taxa list (.csv) the project runs instead of the
country geofence (AddaxAI#136). Nullable, and NULL is today's behaviour,
so the upgrade is a no-op for every current project.

Idempotent: skips the ADD COLUMN step if the column already exists, so a
half-applied attempt (column added, alembic_version not updated) does not
loop on startup with a duplicate-column error.

Revision ID: fa19b0f2a459
Revises: c4e6a8b0d2f3
Create Date: 2026-11-24 13:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fa19b0f2a459"
down_revision: str | None = "c4e6a8b0d2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _projects_has_column(name: str) -> bool:
    inspector = inspect(op.get_bind())
    return any(c["name"] == name for c in inspector.get_columns("projects"))


def upgrade() -> None:
    if not _projects_has_column("taxa_mapping"):
        op.add_column("projects", sa.Column("taxa_mapping", sa.JSON(), nullable=True))


def downgrade() -> None:
    if _projects_has_column("taxa_mapping"):
        op.drop_column("projects", "taxa_mapping")
