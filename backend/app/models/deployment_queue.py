"""
Deployment Queue model - queue entries for batch deployment processing.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` / `processed_at_utc` are tz-aware UTC audit timestamps.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .project import Project
    from .site import Site


QueueStatus = Literal["pending", "processing", "completed", "failed"]


class DeploymentQueue(Base):
    """
    Queue entry for deployment processing.

    Stores user configuration for creating a deployment and running
    ML models. Processed sequentially when user clicks "Process Queue".
    """

    __tablename__ = "deployment_queue"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    # Step 1: Data
    folder_path: Mapped[str] = mapped_column(Text, nullable=False)
    video_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Step 2: Deployment
    site_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("sites.id", ondelete="SET NULL"), nullable=True
    )

    # Datetime offset (seconds) to add to all file timestamps during analysis.
    # NULL = no adjustment. Set by the user in the "Adjust dates" modal when
    # camera firmware had an incorrect clock (factory reset, AM/PM error, etc.).
    datetime_offset_seconds: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    # Optional metadata entered during deployment creation, carried over to
    # the final Deployment record when the queue entry is processed.
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Model configuration now inherited from project (not per-deployment)

    # Processing status
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending, processing, completed, failed
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    processed_at_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Result - created deployment ID after processing
    deployment_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )  # FK not enforced to avoid circular dependencies

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="deployment_queue")
    site: Mapped["Site | None"] = relationship("Site")

    def __repr__(self) -> str:
        return (
            f"<DeploymentQueue(id={self.id}, project_id={self.project_id}, status={self.status})>"
        )
