"""add_classification_fields_to_detections

Revision ID: 08c3ad432da3
Revises: 5cac8bb39056
Create Date: 2026-01-04 16:08:40.165039

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '08c3ad432da3'
down_revision: Union[str, None] = '5cac8bb39056'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add classification_all_probs field to detections table."""
    # Note: species and species_confidence already exist from previous migration
    # Only add the new classification_all_probs field
    op.add_column('detections', sa.Column('classification_all_probs', sa.JSON, nullable=True))


def downgrade() -> None:
    """Remove classification_all_probs field from detections table."""
    # Only drop the new field (species and species_confidence managed by other migration)
    op.drop_column('detections', 'classification_all_probs')
