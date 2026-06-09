"""
Event model - time-clustered groups of files.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `event_start_local` / `event_end_local` are naive wall-clock times in
  the project's local camera timezone, derived from File.captured_at_local.
- `created_at_utc` is a tz-aware UTC audit timestamp.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .deployment import Deployment
    from .event_observation import EventObservation
    from .file import File


# Junction table for many-to-many relationship between events and files
event_files = Table(
    "event_files",
    Base.metadata,
    Column("event_id", String(36), ForeignKey("events.id", ondelete="CASCADE"), primary_key=True),
    Column("file_id", String(36), ForeignKey("files.id", ondelete="CASCADE"), primary_key=True),
    Column("sequence_number", Integer, nullable=True),  # Order within event
    Index("idx_event_files_event", "event_id"),
    Index("idx_event_files_file", "file_id"),
)


class Event(Base):
    """
    Event - time-clustered group of files.

    Events are automatically derived from file timestamps.
    Files within a configurable time threshold are grouped into the same event.
    """

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    deployment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("deployments.id", ondelete="CASCADE"), nullable=False
    )
    event_start_local: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    event_end_local: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Human sign-off that the event's species and counts are correct (set
    # on the Observations page). Distinct from Detection.verified ("a
    # label was confirmed", set on the Labels page). Cleared automatically
    # when the event's species/count set changes (see crud/event_observation).
    verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0", default=False
    )
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    deployment: Mapped["Deployment"] = relationship("Deployment", back_populates="events")
    files: Mapped[list["File"]] = relationship(
        "File", secondary=event_files, back_populates="events"
    )
    observations: Mapped[list["EventObservation"]] = relationship(
        "EventObservation",
        back_populates="event",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("idx_events_deployment", "deployment_id"),
        Index("idx_events_local", "event_start_local", "event_end_local"),
    )

    def __repr__(self) -> str:
        return (
            f"<Event(id={self.id}, event_start_local={self.event_start_local}, "
            f"file_count={self.file_count})>"
        )
