"""Recompute video observation_type from the best frame only

A video is written to disk as a single frame, and everything the user can
see or act on already comes from that frame: the Labels grid, the MaxN
counts, the crops, the embeddings, the annotated still, and the folder the
file is copied into. ``File.observation_type`` was the one thing derived
from every frame, so a video could be summarised by a box that has no
picture anywhere and that the user therefore cannot correct.

The rule is now the same one ``ml/detection_visibility.py`` applies
everywhere else: a video detection counts when it sits on the best frame,
or when a human verified it (a human decision must never end up out of
reach). Images are unchanged.

The column is denormalised and nothing recomputes it on its own, so it
needs this backfill.

**Scoped to videos.** Images are excluded by the outer WHERE rather than
by an extra arm in the subquery. That makes the NULL case structurally
impossible instead of merely handled: an image has both
``frame_number IS NULL`` and ``best_frame_number IS NULL``, and
``NULL = NULL`` is NULL in SQL, so an unscoped version would rely on a
third arm nobody may later "tidy away". It is also provably a no-op for
images and saves a full-table pass on a large library.

A video with ``best_frame_number IS NULL`` (failed frame extraction,
legacy row) has no visible surface at all, so it reads ``blank`` unless
something on it was verified. That matches ``on_visible_frame_of`` and
``calculate_max_n_for_event``, which already treat those videos this way.

Written as correlated subqueries rather than ``UPDATE ... FROM`` (needs
SQLite 3.33) or a CTE, so it runs on every SQLite the app ships with.

Revision ID: 6f7a8b9c0d1e
Revises: 5e6f7a8b9c0d
Create Date: 2026-08-03 12:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6f7a8b9c0d1e"
down_revision: str | None = "5e6f7a8b9c0d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# The deciding detection on a video's visible surface: passing (over its
# project's threshold, or verified) AND on the best frame (or verified),
# strongest first. The trailing `category` in the ORDER BY only makes an
# exact tie deterministic, matching the Python.
_STRONGEST_VISIBLE_CATEGORY = """
    SELECT d.category
    FROM detections d
    WHERE d.file_id = files.id
      AND (
        d.verified = 1
        OR d.confidence >= (
            SELECT p.counting_threshold
            FROM deployments dep
            JOIN projects p ON p.id = dep.project_id
            WHERE dep.id = files.deployment_id
        )
      )
      AND (d.verified = 1 OR d.frame_number = files.best_frame_number)
    ORDER BY d.verified DESC, d.confidence DESC, d.category ASC
    LIMIT 1
"""

# The rule this replaces, for the downgrade: the same strongest-detection
# pick without the frame gate. Duplicated from 5e6f7a8b9c0d on purpose --
# an alembic revision has to be self-contained, so importing it across
# revision modules would be worse than repeating it.
_STRONGEST_CATEGORY = """
    SELECT d.category
    FROM detections d
    WHERE d.file_id = files.id
      AND (
        d.verified = 1
        OR d.confidence >= (
            SELECT p.counting_threshold
            FROM deployments dep
            JOIN projects p ON p.id = dep.project_id
            WHERE dep.id = files.deployment_id
        )
      )
    ORDER BY d.verified DESC, d.confidence DESC, d.category ASC
    LIMIT 1
"""


def _has_files_table(bind) -> bool:
    return "files" in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_files_table(bind):
        return
    op.execute(
        f"UPDATE files SET observation_type = "
        f"COALESCE(({_STRONGEST_VISIBLE_CATEGORY}), 'blank') "
        f"WHERE file_type = 'video'"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_files_table(bind):
        return
    op.execute(
        f"UPDATE files SET observation_type = "
        f"COALESCE(({_STRONGEST_CATEGORY}), 'blank') "
        f"WHERE file_type = 'video'"
    )
