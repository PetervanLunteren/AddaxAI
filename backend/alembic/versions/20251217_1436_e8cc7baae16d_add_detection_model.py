"""add_detection_model

Revision ID: e8cc7baae16d
Revises: 59fe9613a26f
Create Date: 2025-12-17 14:36:33.083538

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8cc7baae16d'
down_revision: Union[str, None] = '59fe9613a26f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create detections table
    op.create_table(
        'detections',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('file_id', sa.String(length=36), nullable=False),
        sa.Column('job_id', sa.String(length=36), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('bbox_x', sa.Float(), nullable=False),
        sa.Column('bbox_y', sa.Float(), nullable=False),
        sa.Column('bbox_width', sa.Float(), nullable=False),
        sa.Column('bbox_height', sa.Float(), nullable=False),
        sa.Column('species', sa.String(length=100), nullable=True),
        sa.Column('species_confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['file_id'], ['files.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_detections_file', 'detections', ['file_id'])
    op.create_index('idx_detections_job', 'detections', ['job_id'])
    op.create_index('idx_detections_category', 'detections', ['category'])
    op.create_index('idx_detections_confidence', 'detections', ['confidence'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_detections_confidence', table_name='detections')
    op.drop_index('idx_detections_category', table_name='detections')
    op.drop_index('idx_detections_job', table_name='detections')
    op.drop_index('idx_detections_file', table_name='detections')

    # Drop table
    op.drop_table('detections')
