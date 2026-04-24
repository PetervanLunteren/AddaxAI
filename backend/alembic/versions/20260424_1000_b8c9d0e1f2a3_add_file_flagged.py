"""add flagged column to files

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-04-24 10:00:00.000000

Adds a user-set flag to the File model, independent of verification.
Mirrors the shape of File.verified / verified_at_utc. A flagged file
is one a user wants to revisit; the flag survives verification and
has its own filter on Events and Files verify tabs.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: Union[str, None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "files",
        sa.Column(
            "flagged",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "files",
        sa.Column(
            "flagged_at_utc",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_index("idx_files_flagged", "files", ["flagged"])


def downgrade() -> None:
    op.drop_index("idx_files_flagged", table_name="files")
    op.drop_column("files", "flagged_at_utc")
    op.drop_column("files", "flagged")
