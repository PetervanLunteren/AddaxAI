"""
Detection model - ML detection results (bounding boxes).

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` / `verified_at_utc` are tz-aware UTC audit timestamps.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .detection_embedding import DetectionEmbedding
    from .file import File
    from .job import Job
    from .label_taxonomy import LabelTaxonomy


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

    # Detection bounding box (normalized coordinates 0-1). Nullable
    # because event-level observations — species seen in a video clip
    # without a frame-anchored ROI — have no spatial annotation. The
    # four columns are required to be all-set or all-null in practice
    # (Pydantic + a CHECK constraint below enforce this); we cannot
    # express the joint nullability in SQLAlchemy's column types alone.
    category: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "animal", "person", "vehicle"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 - 1.0
    bbox_x: Mapped[float | None] = mapped_column(Float, nullable=True)  # Top-left X
    bbox_y: Mapped[float | None] = mapped_column(Float, nullable=True)  # Top-left Y
    bbox_width: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_height: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Classification results (filled by classification models)
    label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    label_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Two precomputed display names, both filled by resolve_label_names at
    # write time. common_name is the SpeciesNet common name (or the cleaned
    # class label); scientific_name is the Latin form ("P. pardus"). The UI
    # picks one via a per-user preference; both degrade to the capitalised
    # category for unclassified detections.
    common_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    scientific_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # The machine's final label (after exclusion, geofence rollup, and
    # smoothing) = exactly what the UI shows before any human relabel.
    # Written by the machine pipeline (JSON load, then postprocessing);
    # never touched by human relabel or verify, so on a verified/relabelled
    # detection it preserves what the AI said. Hence for a non-verified
    # detection original_label == label, and the "ai_classification_label"
    # export column shows the surfaced call, not the raw pre-rollup one
    # (that stays only in results.json on disk).
    # NULL for detector-only projects, person/vehicle detections,
    # and detections analysed before this column existed.
    original_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    original_label_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # FK to LabelTaxonomy — source of truth for taxonomy lookups (exports, filter tree).
    # Nullable: existing detections won't have it, and inference creates detections
    # before taxonomy rows exist. Linked during postprocessing.
    label_taxonomy_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("label_taxonomy.id", ondelete="SET NULL"), nullable=True
    )

    # Classification method (Camtrap DP classificationMethod) - "machine" or "human"
    classification_method: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Video-specific field (None for images, frame index for videos)
    frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Detection-level verification
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_at_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Suggestions review: when True the user clicked "Dismiss" on the
    # cohort this detection belonged to. Dismissed detections are skipped
    # when grouping suggestion cohorts (the toolbar pill, the cohort
    # divider, and the suggestions-sort grid order) but are otherwise
    # untouched — their label and verified state are unchanged, and they
    # still participate as embedding neighbours for other detections'
    # suggestions. A human decision: never reset by reprocessing,
    # smoothing, rollup, or re-embedding.
    suggestion_dismissed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    file: Mapped["File"] = relationship("File", back_populates="detections")
    job: Mapped["Job | None"] = relationship("Job")
    label_taxonomy: Mapped["LabelTaxonomy | None"] = relationship(
        "LabelTaxonomy", back_populates="detections"
    )
    embeddings: Mapped[list["DetectionEmbedding"]] = relationship(
        "DetectionEmbedding", back_populates="detection", cascade="all, delete-orphan"
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_detections_file", "file_id"),
        Index("idx_detections_job", "job_id"),
        Index("idx_detections_category", "category"),
        Index("idx_detections_confidence", "confidence"),
        Index("idx_detections_label", "label"),
        Index("idx_detections_label_confidence", "label_confidence"),
        Index("idx_detections_label_taxonomy", "label_taxonomy_id"),
        Index("idx_detections_frame_number", "frame_number"),
        Index("idx_detections_verified", "verified"),
        Index("idx_detections_original_label", "original_label"),
    )

    def __repr__(self) -> str:
        return (
            f"<Detection(id={self.id}, category={self.category}, "
            f"confidence={self.confidence:.2f})>"
        )
