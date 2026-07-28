"""event counts and verified

Add the human-authoritative count layer and the explicit event sign-off:

- ``event_observations.human_count`` (nullable int): overrides the AI
  MaxN for stats/exports; null means "use max_n".
- ``events.verified`` (bool): explicit human sign-off on the event's
  species and counts (the Observations page), distinct from
  ``detections.verified`` (the Labels page).

Data rollover (replaces the retired box-less "fake frame" flow):

1. Backfill ``events.verified`` from the previous derived rule (an event
   was verified when all its MaxN-frame files were verified, or for blank
   events when any file was verified), so existing sign-offs survive.
2. Backfill ``event_observations.human_count = max_n`` for every species
   that had box-less verified detections (bbox NULL + verified) in the
   event, so the counts the user already entered survive the deletion.
3. Delete the box-less detection rows (all bbox_* NULL). These were the
   per-individual "fake frame" observations, now replaced by human_count.
   Irreversible (restore from a DB backup if needed).

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are each
skipped when the live schema is already in the target state.

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-06-09 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f2a3b4c5d6e7"
down_revision: str | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()

    if "human_count" not in _columns(bind, "event_observations"):
        op.add_column(
            "event_observations",
            sa.Column("human_count", sa.Integer(), nullable=True),
        )

    if "verified" not in _columns(bind, "events"):
        op.add_column(
            "events",
            sa.Column(
                "verified",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            ),
        )

    # 1. Backfill events.verified from the previous derived rule.
    op.execute(
        """
        UPDATE events SET verified = 1
        WHERE (
            EXISTS (
                SELECT 1 FROM event_observations eo
                WHERE eo.event_id = events.id
                  AND eo.max_n_file_id IS NOT NULL
            )
            AND NOT EXISTS (
                SELECT 1 FROM event_observations eo
                JOIN files f ON f.id = eo.max_n_file_id
                WHERE eo.event_id = events.id
                  AND eo.max_n_file_id IS NOT NULL
                  AND f.verified = 0
            )
        ) OR (
            NOT EXISTS (
                SELECT 1 FROM event_observations eo
                WHERE eo.event_id = events.id
                  AND eo.max_n_file_id IS NOT NULL
            )
            AND EXISTS (
                SELECT 1 FROM event_files ef
                JOIN files f ON f.id = ef.file_id
                WHERE ef.event_id = events.id AND f.verified = 1
            )
        )
        """
    )

    # 2. Preserve user-entered counts: where a species had box-less
    #    verified detections, its max_n already includes them, so copy
    #    max_n into human_count before the rows are deleted.
    op.execute(
        """
        UPDATE event_observations
        SET human_count = max_n
        WHERE EXISTS (
            SELECT 1 FROM detections d
            JOIN files f ON f.id = d.file_id
            JOIN event_files ef ON ef.file_id = f.id
            WHERE ef.event_id = event_observations.event_id
              AND d.bbox_x IS NULL
              AND d.verified = 1
              AND (
                (event_observations.label_taxonomy_id IS NOT NULL
                 AND d.label_taxonomy_id = event_observations.label_taxonomy_id)
                OR (event_observations.label_taxonomy_id IS NULL
                    AND event_observations.label IS d.label
                    AND event_observations.category = d.category)
              )
        )
        """
    )

    # 3. Delete the box-less "fake frame" detections (all bbox_* NULL).
    op.execute("DELETE FROM detections WHERE bbox_x IS NULL")


def downgrade() -> None:
    bind = op.get_bind()
    if "human_count" in _columns(bind, "event_observations"):
        op.execute("ALTER TABLE event_observations DROP COLUMN human_count")
    if "verified" in _columns(bind, "events"):
        op.execute("ALTER TABLE events DROP COLUMN verified")
