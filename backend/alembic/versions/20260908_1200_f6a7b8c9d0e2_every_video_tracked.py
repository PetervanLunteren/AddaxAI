"""Every video is tracked

Three things, one step:

1. `projects.video_tracking` goes. Tracking is no longer a switch: every
   video runs through the tracking script.
2. `tracks.frame_path` becomes `tracks.crop_path`. The column used to
   name a full JPEG of the track's representative frame; it now names
   the crop of the representative box the tracking script cuts while it
   runs. The old values point at full frames, which as crops would give
   a card the whole scene, so they are nulled; those tracks show a plain
   tile until their video is analysed again.
3. Videos analysed before this change get their visible boxes as
   one-frame tracks: every untracked box on the video's best frame, and
   every verified untracked box, becomes its own track (start, end and
   representative frame all that frame, one frame, the box's confidence,
   no crop). So the visibility rule has one clause for videos, a box is
   visible on its track's representative frame, and the old "best frame
   or verified" branch is gone. Unverified boxes off the best frame stay
   as they are: rows nothing shows, like the sub-threshold noise, kept
   so a reprocess of an old run still matches its JSON.

Guarded against drifted beta DBs (DEVELOPERS.md): every statement is
skipped when the live schema is already in the target state. The data
step is pinned by `tests/db/test_migration_data.py`.

Revision ID: f6a7b8c9d0e2
Revises: e5f6a7b8c9d1
Create Date: 2026-09-08 12:00:00

"""
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e2"
down_revision: str | None = "e5f6a7b8c9d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def _legacy_boxes_become_tracks(bind) -> None:
    """One track per visible untracked box of every video."""
    rows = bind.execute(
        sa.text(
            "SELECT d.id, d.file_id, d.frame_number, d.confidence "
            "FROM detections d JOIN files f ON f.id = d.file_id "
            "WHERE f.file_type = 'video' AND d.track_id IS NULL "
            "AND d.frame_number IS NOT NULL "
            "AND (d.frame_number = f.best_frame_number OR d.verified = 1) "
            "ORDER BY d.file_id, d.frame_number, d.id"
        )
    ).all()
    if not rows:
        return
    next_key: dict[str, int] = {
        file_id: int(max_key) + 1
        for file_id, max_key in bind.execute(
            sa.text("SELECT file_id, MAX(track_key) FROM tracks GROUP BY file_id")
        ).all()
    }
    now = datetime.now(UTC).isoformat()
    for det_id, file_id, frame, conf in rows:
        key = next_key.get(file_id, 1)
        next_key[file_id] = key + 1
        track_id = str(uuid.uuid4())
        bind.execute(
            sa.text(
                "INSERT INTO tracks (id, file_id, track_key, start_frame, end_frame, "
                "frame_count, max_confidence, representative_frame_number, "
                "crop_path, created_at_utc) VALUES (:id, :file_id, :key, :frame, "
                ":frame, 1, :conf, :frame, NULL, :now)"
            ),
            {"id": track_id, "file_id": file_id, "key": key, "frame": frame,
             "conf": conf, "now": now},
        )
        bind.execute(
            sa.text("UPDATE detections SET track_id = :track_id WHERE id = :det_id"),
            {"track_id": track_id, "det_id": det_id},
        )


def upgrade() -> None:
    bind = op.get_bind()
    if "video_tracking" in _columns(bind, "projects"):
        op.execute("ALTER TABLE projects DROP COLUMN video_tracking")
    track_columns = _columns(bind, "tracks")
    if "frame_path" in track_columns and "crop_path" not in track_columns:
        op.execute("ALTER TABLE tracks RENAME COLUMN frame_path TO crop_path")
        op.execute("UPDATE tracks SET crop_path = NULL")
    _legacy_boxes_become_tracks(bind)


def downgrade() -> None:
    bind = op.get_bind()
    track_columns = _columns(bind, "tracks")
    if "crop_path" in track_columns and "frame_path" not in track_columns:
        op.execute("ALTER TABLE tracks RENAME COLUMN crop_path TO frame_path")
    if "video_tracking" not in _columns(bind, "projects"):
        op.execute(
            "ALTER TABLE projects ADD COLUMN video_tracking BOOLEAN NOT NULL DEFAULT 0"
        )
    # The one-frame tracks stay: the old rule admits a representative box.
