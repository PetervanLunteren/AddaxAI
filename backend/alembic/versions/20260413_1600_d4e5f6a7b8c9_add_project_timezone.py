"""add project timezone

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-04-13 16:00:00.000000

Adds a required `timezone` column (IANA string) to the projects
table. Existing rows get `'UTC'` as a conservative default, then
the server default is dropped so new INSERTs must provide an
explicit value.

The column is metadata only in this iteration: consumed later by
the suncalc overlay on the activity pattern chart.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the column with a temporary server_default so existing rows
    # get a value. Must use batch_alter_table on SQLite for NOT NULL
    # column addition.
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(
            sa.Column(
                "timezone",
                sa.String(64),
                nullable=False,
                server_default="UTC",
            )
        )

    # Drop the server default so future INSERTs must specify a value.
    with op.batch_alter_table("projects") as batch_op:
        batch_op.alter_column("timezone", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("timezone")
