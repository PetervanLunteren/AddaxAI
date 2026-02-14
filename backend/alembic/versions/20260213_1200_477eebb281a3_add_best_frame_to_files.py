"""add best frame to files

Revision ID: 477eebb281a3
Revises: bc299da8af53
Create Date: 2026-02-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '477eebb281a3'
down_revision: Union[str, None] = 'bc299da8af53'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('files', sa.Column('best_frame_number', sa.Integer(), nullable=True))
    op.add_column('files', sa.Column('best_frame_path', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('files', 'best_frame_path')
    op.drop_column('files', 'best_frame_number')
