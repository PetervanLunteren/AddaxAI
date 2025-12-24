"""add_models_to_projects

Revision ID: cc3dc3ddf432
Revises: 03ddb73f6866
Create Date: 2025-12-24 11:34:35.436251

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cc3dc3ddf432'
down_revision: Union[str, None] = '03ddb73f6866'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add model configuration columns to projects table
    # SQLite limitation: server_default stays on column (can't be removed)
    op.add_column('projects',
        sa.Column('detection_model_id', sa.String(100), nullable=False, server_default='MD5A-0-0'))
    op.add_column('projects',
        sa.Column('classification_model_id', sa.String(100), nullable=True))
    op.add_column('projects',
        sa.Column('taxonomy_config', sa.JSON(), nullable=False, server_default='{}'))


def downgrade() -> None:
    # Remove model configuration columns
    op.drop_column('projects', 'taxonomy_config')
    op.drop_column('projects', 'classification_model_id')
    op.drop_column('projects', 'detection_model_id')
