"""
Site model - camera locations within a project.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` is a tz-aware UTC audit timestamp.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .deployment import Deployment
    from .project import Project


class Site(Base):
    """
    Camera trap site (physical location).

    Sites are specific camera locations within a project.
    Each site can have multiple deployments over time.
    """

    __tablename__ = "sites"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    habitat_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="sites")
    # No cascade: deleting a site nulls out its deployments' site_id
    # (FK is ondelete=SET NULL). Deployments are owned by the project,
    # not by the site, and must survive a site being removed.
    deployments: Mapped[list["Deployment"]] = relationship(
        "Deployment", back_populates="site"
    )

    # Constraints
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_site_name_per_project"),)

    def __repr__(self) -> str:
        return f"<Site(id={self.id}, name={self.name}, project_id={self.project_id})>"
