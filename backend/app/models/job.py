"""
Job model - background task tracking.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` / `started_at_utc` / `completed_at_utc` are tz-aware
  UTC audit timestamps.
"""

import uuid
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

JobType = Literal[
    "deployment_analysis",
    "import",
    "ml_inference",
    "export",
    "event_computation",
    "postprocessing",
    "re_embedding",
    "camtrap_export",
]
JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class Job(Base):
    """
    Background job tracking.

    Jobs track long-running background tasks like file imports,
    ML model inference, and data exports.
    """

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # JobType
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # JobStatus
    progress_current: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True
    )  # Job-specific parameters
    result: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True
    )  # Job output/results
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    started_at_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Indexes
    __table_args__ = (
        Index("idx_jobs_status", "status"),
        Index("idx_jobs_created", "created_at_utc"),
    )

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, type={self.type}, status={self.status})>"
