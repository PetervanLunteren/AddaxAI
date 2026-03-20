"""Drop representative_file_id from events table.

Representative files are replaced by MaxN frames derived from
EventObservation rows. Thumbnails and verification now use MaxN frames.

Revision ID: i9j0k1l2m3n4
Revises: h8i9j0k1l2m3
Create Date: 2026-03-20 14:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "i9j0k1l2m3n4"
down_revision: Union[str, None] = "h8i9j0k1l2m3"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_column("representative_file_id")


def downgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.add_column(
            sa.Column("representative_file_id", sa.String(36), nullable=True)
        )
