"""species common + scientific names

Rename the single computed name column ``display_name`` to
``scientific_name`` and add a new ``common_name`` column on both
``detections`` and ``label_taxonomy``. The UI picks one of the two via a
per-user preference; both are precomputed at write time so toggling needs
no refetch and exports stay stable.

``scientific_name`` keeps the existing values verbatim (a pure rename).
``common_name`` is backfilled from the cleaned class label (underscores to
spaces, first letter capitalised), which already degrades to the Latin
taxon where SpeciesNet had no common name. Unclassified detections fall
back to the capitalised category, matching how ``scientific_name`` already
renders them.

Guarded against drifted beta DBs (DEVELOPERS.md): the rename and add are
each skipped when the live schema is already in the target state.

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-06-04 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d0e1f2a3b4"
down_revision: str | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("detections", "label_taxonomy")

# Underscores to spaces, capitalise the first character. Mirrors
# format_common_name() / the frontend normalizeLabel so backfilled values
# match what the runtime writes for new data.
def _clean_sql(col: str) -> str:
    return (
        f"UPPER(SUBSTR(REPLACE({col}, '_', ' '), 1, 1)) "
        f"|| SUBSTR(REPLACE({col}, '_', ' '), 2)"
    )


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    for table in _TABLES:
        cols = _columns(bind, table)
        if "display_name" in cols and "scientific_name" not in cols:
            op.execute(
                f"ALTER TABLE {table} "
                f"RENAME COLUMN display_name TO scientific_name"
            )
        if "common_name" not in _columns(bind, table):
            op.add_column(
                table, sa.Column("common_name", sa.String(100), nullable=True)
            )

    # Backfill common_name from the existing label / category.
    op.execute(
        f"UPDATE label_taxonomy SET common_name = {_clean_sql('name')} "
        f"WHERE common_name IS NULL"
    )
    op.execute(
        "UPDATE detections SET common_name = CASE "
        f"WHEN label IS NOT NULL AND label != '' THEN {_clean_sql('label')} "
        f"ELSE {_clean_sql('category')} END "
        "WHERE common_name IS NULL"
    )


def downgrade() -> None:
    bind = op.get_bind()
    for table in _TABLES:
        cols = _columns(bind, table)
        if "common_name" in cols:
            op.execute(f"ALTER TABLE {table} DROP COLUMN common_name")
        if "scientific_name" in cols and "display_name" not in cols:
            op.execute(
                f"ALTER TABLE {table} "
                f"RENAME COLUMN scientific_name TO display_name"
            )
