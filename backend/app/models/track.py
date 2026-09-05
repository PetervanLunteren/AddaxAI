"""
Track model: one animal followed through the sampled frames of a video.

Written by the tracker (BoT-SORT in the tracking script) when a project
runs with `video_tracking` on. A track is the unit of review for a long
video: the Labels page shows one card per track, at the frame of the
track's highest-confidence box (`representative_frame_number`), and X,
relabel and verify on that card reach every box of the track.

The track holds no label and no verdict of its own. Its label is the
label of its boxes, read from the representative box, and its verdict is
`Detection.verified` on those boxes. One home for each fact, so the
per-frame queries (MaxN, exports, the video overlay) keep working
unchanged and nothing can drift between a track and its boxes.

No foreign key to a representative detection, so there is no cycle
between this table and `detections`: the representative box is the
detection with this track's id and `frame_number ==
representative_frame_number`, and a track holds one box per frame by
construction.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` is a tz-aware UTC audit timestamp.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .detection import Detection
    from .file import File


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    # The tracker's own id, unique within the video (1, 2, 3, ...). Matches
    # `track_id` on the boxes in the run's JSON, so a re-ingest can find
    # the row again.
    track_key: Mapped[int] = mapped_column(Integer, nullable=False)
    # Absolute frame indices in the source video, like Detection.frame_number.
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    # How many sampled frames carry a box of this track.
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # The frame of the track's highest-confidence box: the picture the card
    # shows and the still that is written for it (SharkTrack's choice).
    representative_frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # The JPEG of that frame under the deployment's video_frames folder,
    # written by the same pass that writes the best frame. NULL when the
    # frame could not be decoded; the card then renders without a picture.
    frame_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    file: Mapped["File"] = relationship("File", back_populates="tracks")
    # No delete-orphan: boxes outlive their track row only in the purge,
    # where both are bulk deleted, and the FK sets track_id NULL otherwise.
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="track", passive_deletes=True
    )

    __table_args__ = (
        UniqueConstraint("file_id", "track_key", name="uq_tracks_file_key"),
        Index("idx_tracks_file", "file_id"),
    )

    @property
    def has_frame(self) -> bool:
        """Whether the representative frame's JPEG was written."""
        return self.frame_path is not None

    def __repr__(self) -> str:
        return (
            f"<Track(id={self.id}, file_id={self.file_id}, key={self.track_key}, "
            f"frames={self.start_frame}-{self.end_frame})>"
        )
