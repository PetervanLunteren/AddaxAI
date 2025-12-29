"""add_taxonomic_aggregation_threshold

Revision ID: 353fbc92fc68
Revises: acc38a37a0ba
Create Date: 2025-12-29 09:19:38.252079

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '353fbc92fc68'
down_revision: Union[str, None] = 'acc38a37a0ba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('taxonomic_aggregation_threshold', sa.Float(), nullable=False, server_default='0.65'))


def downgrade() -> None:
    op.drop_column('projects', 'taxonomic_aggregation_threshold')
