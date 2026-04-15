"""
Detection embedding model - stores DINOv2 feature vectors for detections.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` is a tz-aware UTC audit timestamp.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .detection import Detection


class DetectionEmbedding(Base):
    """
    Embedding vector for a detection crop.

    Stores float16 DINOv2 embeddings as raw bytes.
    Each detection can have one embedding per model.
    """

    __tablename__ = "detection_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    detection_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("detections.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("jobs.id"), nullable=True)
    embedding_model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # float16 bytes
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    l2_norm: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # pre-computed for cosine similarity
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    detection: Mapped["Detection"] = relationship("Detection", back_populates="embeddings")

    # Indexes
    __table_args__ = (
        Index("idx_detection_embeddings_detection", "detection_id"),
        Index("idx_detection_embeddings_model", "embedding_model_id"),
        Index(
            "idx_detection_embeddings_detection_model",
            "detection_id",
            "embedding_model_id",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<DetectionEmbedding(id={self.id}, detection_id={self.detection_id}, "
            f"model={self.embedding_model_id}, dim={self.dimension})>"
        )
