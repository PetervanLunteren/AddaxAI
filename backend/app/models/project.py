"""
Project model - top level container for camera trap projects.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Clear, explicit relationships
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .deployment_queue import DeploymentQueue
    from .site import Site


class Project(Base):
    """
    Camera trap project.

    A project is the top-level organizational unit containing sites,
    deployments, and all associated media files and detections.
    """

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Model configuration (project-scoped)
    detection_model_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default="MD5A-0-0"
    )
    classification_model_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    excluded_classes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    # SpeciesNet geographic location (alternative to excluded_classes)
    country_code: Mapped[str | None] = mapped_column(
        String(3), nullable=True
    )
    state_code: Mapped[str | None] = mapped_column(
        String(2), nullable=True
    )

    # Detection and processing settings
    detection_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5
    )
    event_smoothing: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    taxonomic_rollup: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    taxonomic_rollup_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.65
    )
    independence_interval: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1800  # seconds
    )

    # Relationships
    sites: Mapped[list["Site"]] = relationship(
        "Site", back_populates="project", cascade="all, delete-orphan"
    )
    deployment_queue: Mapped[list["DeploymentQueue"]] = relationship(
        "DeploymentQueue", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name})>"
