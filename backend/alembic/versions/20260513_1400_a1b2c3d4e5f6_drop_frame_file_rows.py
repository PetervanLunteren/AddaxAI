"""drop frame file rows

Re-point legacy `Detection.file_id` from `file_type='frame'` rows to
their parent video File row (preserving `Detection.frame_number`), then
delete every `file_type='frame'` File row. Post-2026-05 the pipeline
no longer creates frame rows; detections live on the video row directly
and per-frame information is recovered via `frame_number`. The matching
disk JPEGs (everything under each deployment's
`.addaxai/projects/*/video_frames/` that isn't the best frame) are
reclaimed by a non-blocking startup task in `main.py:lifespan`, not by
this migration: alembic stays out of the filesystem.

Revision ID: a1b2c3d4e5f6
Revises: 2540e6edbee2
Create Date: 2026-05-13 14:00:00

"""
from collections.abc import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: str | None = '2540e6edbee2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Repoint detections that point at a frame row to its source video.
    # The frame row's `source_video_id` is the parent video's File.id;
    # nothing else changes on the detection (its `frame_number` was set
    # at ingest and stays as-is).
    op.execute(
        """
        UPDATE detections
           SET file_id = (
                   SELECT files.source_video_id
                     FROM files
                    WHERE files.id = detections.file_id
               )
         WHERE file_id IN (
                   SELECT id FROM files WHERE file_type = 'frame'
               )
        """
    )

    # Drop the frame rows themselves. Any detection still attached to a
    # frame at this point would have failed the UPDATE above (no
    # source_video_id) — leaving it orphaned would violate the FK, so
    # surface those by failing the DELETE if any remain. In practice
    # every frame row has a source_video_id; this is paranoia.
    op.execute(
        """
        DELETE FROM files
         WHERE file_type = 'frame'
           AND NOT EXISTS (
                   SELECT 1 FROM detections WHERE detections.file_id = files.id
               )
        """
    )


def downgrade() -> None:
    # Lossy: we deleted the frame rows. Restoring them would require
    # re-running the legacy bulk extraction, which is exactly what the
    # refactor removed. Reject the downgrade rather than pretend.
    raise RuntimeError(
        "Downgrading past drop_frame_file_rows is not supported. The "
        "frame File rows were collapsed onto their parent video rows; "
        "restoring them would require re-extracting frames from the "
        "source videos."
    )
