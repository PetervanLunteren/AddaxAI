"""replace_taxonomy_config_with_excluded_classes_and_add_processing_settings

Revision ID: acc38a37a0ba
Revises: 3ee4969ac7fb
Create Date: 2025-12-28 09:31:46.376767

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'acc38a37a0ba'
down_revision: Union[str, None] = '3ee4969ac7fb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old taxonomy_config column
    op.drop_column('projects', 'taxonomy_config')

    # Add new excluded_classes column (JSON array)
    op.add_column('projects', sa.Column('excluded_classes', sa.JSON(), nullable=False, server_default='[]'))

    # Add new processing settings columns
    op.add_column('projects', sa.Column('detection_threshold', sa.Float(), nullable=False, server_default='0.5'))
    op.add_column('projects', sa.Column('event_smoothing', sa.Boolean(), nullable=False, server_default='1'))
    op.add_column('projects', sa.Column('taxonomic_aggregation', sa.Boolean(), nullable=False, server_default='1'))
    op.add_column('projects', sa.Column('independence_interval', sa.Integer(), nullable=False, server_default='1800'))


def downgrade() -> None:
    # Remove new columns
    op.drop_column('projects', 'independence_interval')
    op.drop_column('projects', 'taxonomic_aggregation')
    op.drop_column('projects', 'event_smoothing')
    op.drop_column('projects', 'detection_threshold')
    op.drop_column('projects', 'excluded_classes')

    # Restore old taxonomy_config column
    op.add_column('projects', sa.Column('taxonomy_config', sa.JSON(), nullable=False, server_default='{}'))
