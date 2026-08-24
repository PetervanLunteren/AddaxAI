"""add taxon_variant to label_taxonomy

One optional rank below species for classification models whose classes
sit deeper than a species (adult vs juvenile fox). Rows with a non-empty
variant carry level="variant"; every existing row is untouched (NULL).

Guarded with a presence check so a drifted DB that somehow already has
the column turns into a clean no-op instead of a mid-chain crash (see
DEVELOPERS.md "Guard DDL anyway in new migrations").

Revision ID: 8b9c0d1e2f3a
Revises: 7a8b9c0d1e2f
Create Date: 2026-08-24 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8b9c0d1e2f3a"
down_revision: str | None = "7a8b9c0d1e2f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _label_taxonomy_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("label_taxonomy")}


def upgrade() -> None:
    bind = op.get_bind()
    if "taxon_variant" not in _label_taxonomy_columns(bind):
        op.add_column(
            "label_taxonomy",
            sa.Column("taxon_variant", sa.String(100), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "taxon_variant" in _label_taxonomy_columns(bind):
        op.execute("ALTER TABLE label_taxonomy DROP COLUMN taxon_variant")
