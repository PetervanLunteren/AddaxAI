"""Add display_name to label_taxonomy and backfill.

Stores pre-computed display names (e.g. "G. camelopardalis", "Felidae")
on the label_taxonomy table as the single source of truth. Also backfills
Detection.display_name from label_taxonomy for any existing NULL values.

Revision ID: k1l2m3n4o5p6
Revises: i9j0k1l2m3n4
Create Date: 2026-03-27 10:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "k1l2m3n4o5p6"
down_revision: Union[str, None] = "i9j0k1l2m3n4"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    # 1. Add display_name column to label_taxonomy
    with op.batch_alter_table("label_taxonomy") as batch_op:
        batch_op.add_column(
            sa.Column("display_name", sa.String(100), nullable=True)
        )

    # 2. Backfill label_taxonomy.display_name from taxonomy fields.
    #    Logic mirrors format_display_name_from_taxonomy_row():
    #    - Species with genus: "G. epithet"
    #    - Genus only: "Genus" (capitalized)
    #    - Family/Order/Class: capitalized
    #    - Fallback: capitalized name
    op.execute(
        sa.text("""
            UPDATE label_taxonomy SET display_name = CASE
                WHEN taxon_species IS NOT NULL AND taxon_genus IS NOT NULL
                    THEN UPPER(SUBSTR(taxon_genus, 1, 1)) || '. ' || taxon_species
                WHEN taxon_genus IS NOT NULL
                    THEN UPPER(SUBSTR(taxon_genus, 1, 1))
                         || LOWER(SUBSTR(taxon_genus, 2))
                WHEN taxon_family IS NOT NULL
                    THEN UPPER(SUBSTR(taxon_family, 1, 1))
                         || LOWER(SUBSTR(taxon_family, 2))
                WHEN taxon_order IS NOT NULL
                    THEN UPPER(SUBSTR(taxon_order, 1, 1))
                         || LOWER(SUBSTR(taxon_order, 2))
                WHEN taxon_class IS NOT NULL
                    THEN UPPER(SUBSTR(taxon_class, 1, 1))
                         || LOWER(SUBSTR(taxon_class, 2))
                ELSE UPPER(SUBSTR(name, 1, 1))
                     || SUBSTR(name, 2)
            END
            WHERE display_name IS NULL
        """)
    )

    # 3. Backfill Detection.display_name from label_taxonomy.display_name
    #    for detections that have a taxonomy link but NULL display_name.
    op.execute(
        sa.text("""
            UPDATE detections SET display_name = (
                SELECT lt.display_name FROM label_taxonomy lt
                WHERE lt.id = detections.label_taxonomy_id
            )
            WHERE label_taxonomy_id IS NOT NULL AND display_name IS NULL
        """)
    )


def downgrade() -> None:
    with op.batch_alter_table("label_taxonomy") as batch_op:
        batch_op.drop_column("display_name")
