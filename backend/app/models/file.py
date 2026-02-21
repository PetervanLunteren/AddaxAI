"""
File model - images and videos from camera traps.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit constraints
- No silent failures
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import ColumnElement
from sqlalchemy.sql.schema import ForeignKey

from app.db.base import Base

if TYPE_CHECKING:
    from .deployment import Deployment
    from .detection import Detection
    from .event import Event


FileType = Literal["image", "video", "frame"]


class File(Base):
    """
    Media file (image or video) from a camera trap.

    Files belong to a deployment and can be grouped into events
    based on temporal clustering.
    """

    __tablename__ = "files"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    deployment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("deployments.id", ondelete="CASCADE"), nullable=False
    )

    # File metadata
    file_path: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Absolute path - same file can be analyzed by multiple projects
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)  # 'image' or 'video'
    file_format: Mapped[str | None] = mapped_column(
        String(10), nullable=True
    )  # 'jpg', 'png', 'mp4', etc.
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Temporal metadata
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False
    )  # From EXIF or filename
    exif_data: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True
    )  # Full EXIF as JSON blob

    # Observation type (Camtrap DP observationType) - summary for filtering/browsing
    # Priority derived from detections: animal > human > vehicle > blank > unclassified
    observation_type: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="unclassified"
    )

    # Video-specific
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Best frame (video only)
    best_frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_frame_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    frame_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Frame-specific (file_type="frame" only)
    source_video_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=True
    )
    source_frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Verification fields
    verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0", default=False
    )
    verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    favorited: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0", default=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    deployment: Mapped["Deployment"] = relationship("Deployment", back_populates="files")
    events: Mapped[list["Event"]] = relationship(
        "Event", secondary="event_files", back_populates="files"
    )
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="file", cascade="all, delete-orphan"
    )
    source_video: Mapped["File | None"] = relationship(
        "File", remote_side="File.id", foreign_keys=[source_video_id]
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_files_deployment", "deployment_id"),
        Index("idx_files_timestamp", "timestamp"),
        Index("idx_files_verified", "verified"),
        Index("idx_files_source_video", "source_video_id"),
    )

    def __repr__(self) -> str:
        return f"<File(id={self.id}, file_path={self.file_path}, timestamp={self.timestamp})>"
