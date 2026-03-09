"""Add species_taxonomy table.

Stores parsed taxonomy data from taxonomy.csv and rolled-up entries,
enabling server-side species filter tree building.

Revision ID: d4e5f6a7b8c0
Revises: c3d4e5f6a7b9
Create Date: 2026-03-06 10:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c0"
down_revision: Union[str, None] = "c3d4e5f6a7b9"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.create_table(
        "species_taxonomy",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("classification_model_id", sa.String(100), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("taxon_class", sa.String(100), nullable=True),
        sa.Column("taxon_order", sa.String(100), nullable=True),
        sa.Column("taxon_family", sa.String(100), nullable=True),
        sa.Column("taxon_genus", sa.String(100), nullable=True),
        sa.Column("taxon_species", sa.String(100), nullable=True),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("is_custom", sa.Boolean, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint(
            "classification_model_id", "name",
            name="uq_species_taxonomy_model_name",
        ),
    )
    op.create_index(
        "idx_species_taxonomy_model", "species_taxonomy", ["classification_model_id"]
    )
    op.create_index(
        "idx_species_taxonomy_name", "species_taxonomy", ["name"]
    )


def downgrade() -> None:
    op.drop_index("idx_species_taxonomy_name", table_name="species_taxonomy")
    op.drop_index("idx_species_taxonomy_model", table_name="species_taxonomy")
    op.drop_table("species_taxonomy")
