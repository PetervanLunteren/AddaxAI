"""rename_taxonomic_aggregation_to_rollup

Revision ID: 761c29fe9d41
Revises: 353fbc92fc68
Create Date: 2025-12-29 12:03:42.570216

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '761c29fe9d41'
down_revision: Union[str, None] = '353fbc92fc68'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename columns in SQLite (rename column is supported in SQLite 3.25+)
    op.alter_column('projects', 'taxonomic_aggregation', new_column_name='taxonomic_rollup')
    op.alter_column('projects', 'taxonomic_aggregation_threshold', new_column_name='taxonomic_rollup_threshold')


def downgrade() -> None:
    # Rename back
    op.alter_column('projects', 'taxonomic_rollup', new_column_name='taxonomic_aggregation')
    op.alter_column('projects', 'taxonomic_rollup_threshold', new_column_name='taxonomic_aggregation_threshold')
