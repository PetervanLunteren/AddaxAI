"""
Deployment model - camera deployment periods at sites.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit relationships
- Clear constraints
"""

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import Date, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .event import Event
    from .file import File
    from .site import Site


FolderStatus = Literal["valid", "missing", "needs_relink"]


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
        String(36), ForeignKey("sites.id", ondelete="CASCADE"), nullable=False
    )

    # File storage
    folder_path: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Absolute path to deployment folder
    folder_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="valid"
    )  # "valid", "missing", "needs_relink"
    last_validated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )  # When folder was last checked

    # Deployment metadata
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    camera_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    camera_serial: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
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
        return f"<Deployment(id={self.id}, site_id={self.site_id}, start_date={self.start_date})>"
