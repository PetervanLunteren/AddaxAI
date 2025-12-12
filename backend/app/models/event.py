"""
Event model - time-clustered groups of files.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Clear relationships
- Explicit indexes
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Table
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .deployment import Deployment
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
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    deployment: Mapped["Deployment"] = relationship("Deployment", back_populates="events")
    files: Mapped[list["File"]] = relationship(
        "File", secondary=event_files, back_populates="events"
    )

    # Indexes
    __table_args__ = (
        Index("idx_events_deployment", "deployment_id"),
        Index("idx_events_time", "start_time", "end_time"),
    )

    def __repr__(self) -> str:
        return f"<Event(id={self.id}, start_time={self.start_time}, file_count={self.file_count})>"
