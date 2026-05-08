"""add observations max detections to project

Revision ID: 03e058c707df
Revises: 9c173fff3bcd
Create Date: 2026-05-08 14:37:42.455208

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "03e058c707df"
down_revision: str | None = "9c173fff3bcd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # server_default backfills existing rows. Future rows take their
    # default from the SQLAlchemy model (20000), so the server default
    # is only load-bearing during this migration.
    op.add_column(
        "projects",
        sa.Column(
            "observations_max_detections",
            sa.Integer(),
            nullable=False,
            server_default="20000",
        ),
    )


def downgrade() -> None:
    op.drop_column("projects", "observations_max_detections")
