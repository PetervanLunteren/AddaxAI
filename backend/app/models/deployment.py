"""
Deployment model - camera deployment periods at sites.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `start_date_local` / `end_date_local` are calendar dates in the
  project's local camera timezone, derived from File.captured_at_local.
- `created_at_utc` / `last_validated_at_utc` are tz-aware UTC audit
  timestamps.
"""

import uuid
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .event import Event
    from .file import File
    from .site import Site


FolderStatus = Literal["valid", "needs_relink"]


class Deployment(Base):
    """
    Camera deployment period at a site.

    Represents a specific time period when a camera was deployed
    at a site. Multiple deployments can occur at the same site
    over time (e.g., camera replaced, repositioned, etc.).
    """

    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    site_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sites.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # File storage
    folder_path: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Absolute path to deployment folder
    folder_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="valid"
    )  # "valid", "needs_relink"
    last_validated_at_utc: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )  # When folder was last checked (UTC)

    # Deployment metadata. Dates are in the project's local camera timezone,
    # derived from File.captured_at_local after analysis.
    start_date_local: Mapped[date] = mapped_column(Date, nullable=False)
    end_date_local: Mapped[date | None] = mapped_column(Date, nullable=True)
    camera_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    camera_serial: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Audit: the datetime offset (seconds) that was applied to all file
    # timestamps when this deployment was analyzed. Informational only,
    # not used in queries. NULL = no offset was applied.
    datetime_offset_seconds: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    site: Mapped["Site"] = relationship("Site", back_populates="deployments")
    files: Mapped[list["File"]] = relationship(
        "File", back_populates="deployment", cascade="all, delete-orphan"
    )
    events: Mapped[list["Event"]] = relationship(
        "Event", back_populates="deployment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Deployment(id={self.id}, site_id={self.site_id}, "
            f"start_date_local={self.start_date_local})>"
        )
