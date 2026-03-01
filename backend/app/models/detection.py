"""
Detection model - ML detection results (bounding boxes).

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit relationships
- Clear indexes
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .detection_embedding import DetectionEmbedding
    from .file import File
    from .job import Job


class Detection(Base):
    """
    ML detection result (bounding box + confidence).

    Represents a single detected object in an image from ML inference.
    Created by detection models (e.g., MegaDetector) and optionally
    enriched with species classification later.
    """

    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("jobs.id"), nullable=True
    )

    # Detection bounding box (normalized coordinates 0-1)
    category: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "animal", "person", "vehicle"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 - 1.0
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)  # Top-left X
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)  # Top-left Y
    bbox_width: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_height: Mapped[float] = mapped_column(Float, nullable=False)

    # Classification results (filled by classification models)
    species: Mapped[str | None] = mapped_column(String(100), nullable=True)
    species_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Classification method (Camtrap DP classificationMethod) - "machine" or "human"
    classification_method: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Video-specific field (None for images, frame index for videos)
    frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Detection-level verification
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    file: Mapped["File"] = relationship("File", back_populates="detections")
    job: Mapped["Job | None"] = relationship("Job")
    embeddings: Mapped[list["DetectionEmbedding"]] = relationship(
        "DetectionEmbedding", back_populates="detection", cascade="all, delete-orphan"
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_detections_file", "file_id"),
        Index("idx_detections_job", "job_id"),
        Index("idx_detections_category", "category"),
        Index("idx_detections_confidence", "confidence"),
        Index("idx_detections_species", "species"),
        Index("idx_detections_species_confidence", "species_confidence"),
        Index("idx_detections_frame_number", "frame_number"),
        Index("idx_detections_verified", "verified"),
    )

    def __repr__(self) -> str:
        return (
            f"<Detection(id={self.id}, category={self.category}, "
            f"confidence={self.confidence:.2f})>"
        )
