"""remove_models_from_deployment_queue

Revision ID: 3ee4969ac7fb
Revises: cc3dc3ddf432
Create Date: 2025-12-24 11:35:05.390296

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ee4969ac7fb'
down_revision: Union[str, None] = 'cc3dc3ddf432'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Remove model selection columns from deployment_queue
    # Models are now project-scoped, not deployment-scoped
    op.drop_column('deployment_queue', 'detection_model_id')
    op.drop_column('deployment_queue', 'classification_model_id')
    op.drop_column('deployment_queue', 'species_list')


def downgrade() -> None:
    # Restore model selection columns (for rollback)
    op.add_column('deployment_queue',
        sa.Column('detection_model_id', sa.String(100), nullable=True))
    op.add_column('deployment_queue',
        sa.Column('classification_model_id', sa.String(100), nullable=True))
    op.add_column('deployment_queue',
        sa.Column('species_list', sa.JSON(), nullable=True))
