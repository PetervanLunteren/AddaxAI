"""Add event_observations table for MaxN counting.

Stores per-species MaxN counts within each event. MaxN is the maximum
number of individuals of a species visible in any single image within
an event, preventing double-counting across frames.

Revision ID: h8i9j0k1l2m3
Revises: g7h8i9j0k1l2
Create Date: 2026-03-20 12:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "h8i9j0k1l2m3"
down_revision: Union[str, None] = "g7h8i9j0k1l2"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.create_table(
        "event_observations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(36),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("max_n", sa.Integer, nullable=False),
        sa.Column(
            "max_n_file_id",
            sa.String(36),
            sa.ForeignKey("files.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.UniqueConstraint("event_id", "label", name="uq_event_obs_event_label"),
    )
    op.create_index("idx_event_obs_event", "event_observations", ["event_id"])
    op.create_index("idx_event_obs_label", "event_observations", ["label"])


def downgrade() -> None:
    op.drop_index("idx_event_obs_label", table_name="event_observations")
    op.drop_index("idx_event_obs_event", table_name="event_observations")
    op.drop_table("event_observations")
