"""Make custom labels global (merge per-project rows into one shared set)

Custom labels used to be scoped to one project (``project_id`` set,
``is_custom=1``). They are now global, shared across every folder run and
project, and live at ``project_id IS NULL``. This migration moves the
existing per-project custom rows into that shared scope.

Per name (case-insensitive), the surviving row is the one with taxonomy
ranks filled, else the oldest. Its ``project_id`` is cleared. Every other
same-name custom row is merged into it: detections and event observations
that pointed at a loser are repointed at the winner, then the losers are
deleted. A lone custom row simply has its ``project_id`` cleared.

Data only, no schema change. Deleting the losers matches zero rows on an
empty or already-global database, so the chain is safe from base. The
merge is not reversible (the per-project origin is not recorded), so
downgrade is a no-op.

Revision ID: c3d4e5f6a7b9
Revises: b2c3d4e5f6a8
Create Date: 2026-09-05 12:00:00

"""
from collections import defaultdict
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b9"
down_revision: str | None = "b2c3d4e5f6a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RANK_COLUMNS = (
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
    "taxon_variant",
)


def _has_ranks(row: sa.engine.Row) -> bool:
    return any(getattr(row, c) is not None for c in _RANK_COLUMNS)


def upgrade() -> None:
    bind = op.get_bind()

    rows = bind.execute(
        sa.text(
            "SELECT id, name, project_id, created_at_utc, "
            + ", ".join(_RANK_COLUMNS)
            + " FROM label_taxonomy WHERE is_custom = 1"
        )
    ).all()

    # Group by case-insensitive name.
    groups: dict[str, list[sa.engine.Row]] = defaultdict(list)
    for row in rows:
        groups[row.name.lower()].append(row)

    for group in groups.values():
        # Winner: a row with ranks beats a bare one; ties break on the
        # oldest created_at_utc, then id for determinism.
        winner = min(
            group,
            key=lambda r: (not _has_ranks(r), r.created_at_utc or "", r.id),
        )
        for loser in group:
            if loser.id == winner.id:
                continue
            for table in ("detections", "event_observations"):
                bind.execute(
                    sa.text(
                        f"UPDATE {table} SET label_taxonomy_id = :win "
                        "WHERE label_taxonomy_id = :lose"
                    ),
                    {"win": winner.id, "lose": loser.id},
                )
            bind.execute(
                sa.text("DELETE FROM label_taxonomy WHERE id = :id"),
                {"id": loser.id},
            )
        if winner.project_id is not None:
            bind.execute(
                sa.text(
                    "UPDATE label_taxonomy SET project_id = NULL WHERE id = :id"
                ),
                {"id": winner.id},
            )


def downgrade() -> None:
    # The per-project origin of each merged row is not recorded, so the
    # merge cannot be undone. The schema is unchanged, so this is a no-op.
    pass
