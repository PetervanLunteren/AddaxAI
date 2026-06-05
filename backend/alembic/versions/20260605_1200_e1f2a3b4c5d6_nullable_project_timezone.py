"""nullable project timezone

Make ``projects.timezone`` nullable. A new project now starts with no
timezone and auto-derives one from its first site's coordinates
(``crud.site.create_site`` + ``utils.timezone_from_coords``); NULL means
"auto / not set yet". Existing projects keep their stored value.

SQLite nullability changes need batch mode (no native single-statement
DDL), per DEVELOPERS.md. Downgrade backfills NULL -> 'UTC' before
restoring NOT NULL so the constraint can be re-applied.

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-06-05 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: str | None = "d0e1f2a3b4c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch:
        batch.alter_column(
            "timezone",
            existing_type=sa.String(64),
            nullable=True,
        )


def downgrade() -> None:
    op.execute("UPDATE projects SET timezone = 'UTC' WHERE timezone IS NULL")
    with op.batch_alter_table("projects") as batch:
        batch.alter_column(
            "timezone",
            existing_type=sa.String(64),
            nullable=False,
        )
