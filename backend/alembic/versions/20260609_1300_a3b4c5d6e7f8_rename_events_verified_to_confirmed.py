"""rename events.verified to events.confirmed

The Observations-page event sign-off is surfaced to users as "Confirm"
(asserting the species + counts are the final record), distinct from the
Labels-page "verify" of a detection's label. Rename the column to match so
code and UI agree: ``events.verified`` -> ``events.confirmed``. All data is
preserved (an in-place column rename).

Guarded against drifted DBs (DEVELOPERS.md): skipped when the live schema
is already in the target state.

Revision ID: a3b4c5d6e7f8
Revises: f2a3b4c5d6e7
Create Date: 2026-06-09 13:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3b4c5d6e7f8"
down_revision: str | None = "f2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    cols = _columns(bind, "events")
    if "verified" in cols and "confirmed" not in cols:
        op.alter_column("events", "verified", new_column_name="confirmed")


def downgrade() -> None:
    bind = op.get_bind()
    cols = _columns(bind, "events")
    if "confirmed" in cols and "verified" not in cols:
        op.alter_column("events", "confirmed", new_column_name="verified")
