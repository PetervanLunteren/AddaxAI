"""
Site model - camera locations within a project.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit foreign key relationships
- No optional fields where not needed
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint
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
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    habitat_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="sites")
    deployments: Mapped[list["Deployment"]] = relationship(
        "Deployment", back_populates="site", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_site_name_per_project"),)

    def __repr__(self) -> str:
        return f"<Site(id={self.id}, name={self.name}, project_id={self.project_id})>"
