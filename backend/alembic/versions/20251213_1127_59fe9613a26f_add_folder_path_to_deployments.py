"""add_folder_path_to_deployments

Revision ID: 59fe9613a26f
Revises: 
Create Date: 2025-12-13 11:27:33.625780

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '59fe9613a26f'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add folder_path, folder_status, and last_validated_at columns to deployments table
    op.add_column('deployments', sa.Column('folder_path', sa.Text(), nullable=True))
    op.add_column('deployments', sa.Column('folder_status', sa.String(length=20), nullable=False, server_default='valid'))
    op.add_column('deployments', sa.Column('last_validated_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    # Remove the added columns
    op.drop_column('deployments', 'last_validated_at')
    op.drop_column('deployments', 'folder_status')
    op.drop_column('deployments', 'folder_path')
