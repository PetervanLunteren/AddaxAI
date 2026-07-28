"""drop observations_max_detections

The Observations-grid similarity-sort cap was a per-project DB column.
It is now a per-user view preference persisted to localStorage in the
Observations tab's view-options popover (next to tile size and label
dividers), so the column is no longer read by any code path. Drop it
to keep the schema honest.

Existing per-project values are lost on upgrade; the frontend default
is the same 20000 the column defaulted to, so nothing visibly changes
unless a user had previously raised the cap and will now have to set
it again in the popover.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-19 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _projects_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("projects")}


def upgrade() -> None:
    # Raw DDL on purpose. The natural alembic equivalent
    # (``op.batch_alter_table("projects") as batch: batch.drop_column(...)``)
    # builds its in-memory copy of the table from target_metadata
    # (the SQLAlchemy models). Since this migration ships alongside
    # removal of the column from ``Project``, the metadata no longer
    # has the column and the batch flush blows up with KeyError on the
    # column it was asked to drop. ``copy_from`` is the documented
    # escape hatch but doesn't reliably pick up the column on every
    # live DB. SQLite 3.35+ (we're on 3.51 via Python 3.13) supports
    # ALTER TABLE DROP COLUMN natively, so we just issue the DDL and
    # skip the recreate dance entirely.
    #
    # Guarded by a presence check so a DB whose stamp disagrees with
    # its live schema (historical stamp_head bug, half-applied
    # migration, hand-edited alembic_version row) doesn't crash the
    # startup. If the column is already gone, leave the DB alone — the
    # end state matches what this migration was trying to achieve.
    bind = op.get_bind()
    if "observations_max_detections" in _projects_columns(bind):
        op.execute(
            "ALTER TABLE projects DROP COLUMN observations_max_detections"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "observations_max_detections" not in _projects_columns(bind):
        op.execute(
            "ALTER TABLE projects "
            "ADD COLUMN observations_max_detections INTEGER NOT NULL "
            "DEFAULT 20000"
        )
