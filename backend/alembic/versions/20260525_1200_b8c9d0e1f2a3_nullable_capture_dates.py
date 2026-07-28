"""nullable capture dates

Allow `files.captured_at_local` and `events.event_start_local` /
`event_end_local` to be NULL so images with no EXIF capture date are
ingested (data-agnostic) instead of dropped. A date-less file becomes
its own single-file event with NULL bounds; time-based stats exclude
the null-date rows and the dashboard surfaces how many lack a date.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-05-25 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8c9d0e1f2a3"
down_revision: str | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("files") as batch_op:
        batch_op.alter_column(
            "captured_at_local", existing_type=sa.DateTime(), nullable=True
        )
    with op.batch_alter_table("events") as batch_op:
        batch_op.alter_column(
            "event_start_local", existing_type=sa.DateTime(), nullable=True
        )
        batch_op.alter_column(
            "event_end_local", existing_type=sa.DateTime(), nullable=True
        )


def downgrade() -> None:
    # Re-applying NOT NULL needs every row populated. There's no real
    # capture date to restore for the date-less rows this feature created,
    # so stamp a meaningless sentinel (matches the bbox-nullable downgrade);
    # the downgrade is lossy.
    op.execute(
        "UPDATE files SET captured_at_local = '1970-01-01 00:00:00' "
        "WHERE captured_at_local IS NULL"
    )
    op.execute(
        "UPDATE events SET event_start_local = '1970-01-01 00:00:00' "
        "WHERE event_start_local IS NULL"
    )
    op.execute(
        "UPDATE events SET event_end_local = '1970-01-01 00:00:00' "
        "WHERE event_end_local IS NULL"
    )
    with op.batch_alter_table("events") as batch_op:
        batch_op.alter_column(
            "event_end_local", existing_type=sa.DateTime(), nullable=False
        )
        batch_op.alter_column(
            "event_start_local", existing_type=sa.DateTime(), nullable=False
        )
    with op.batch_alter_table("files") as batch_op:
        batch_op.alter_column(
            "captured_at_local", existing_type=sa.DateTime(), nullable=False
        )
