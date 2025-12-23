"""add_deployment_queue

Revision ID: 03ddb73f6866
Revises: e8cc7baae16d
Create Date: 2025-12-23 15:38:12.660301

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '03ddb73f6866'
down_revision: Union[str, None] = 'e8cc7baae16d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create deployment_queue table
    op.create_table(
        'deployment_queue',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('project_id', sa.String(length=36), nullable=False),
        sa.Column('folder_path', sa.Text(), nullable=False),
        sa.Column('site_id', sa.String(length=36), nullable=True),
        sa.Column('detection_model_id', sa.String(length=100), nullable=True),
        sa.Column('classification_model_id', sa.String(length=100), nullable=True),
        sa.Column('species_list', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('deployment_id', sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['site_id'], ['sites.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for common queries
    op.create_index('idx_deployment_queue_project', 'deployment_queue', ['project_id'])
    op.create_index('idx_deployment_queue_status', 'deployment_queue', ['status'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_deployment_queue_status', table_name='deployment_queue')
    op.drop_index('idx_deployment_queue_project', table_name='deployment_queue')

    # Drop table
    op.drop_table('deployment_queue')
