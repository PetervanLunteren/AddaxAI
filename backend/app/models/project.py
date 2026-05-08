"""
Project model - top level container for camera trap projects.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` / `updated_at_utc` are tz-aware UTC audit timestamps.
- `timezone` records the wall-clock timezone the cameras were
  configured to; it's metadata used for sun-calc and export, never for
  converting stored datetimes.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .deployment import Deployment
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
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Model configuration (project-scoped)
    detection_model_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default="MD5A-0-0"
    )
    classification_model_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    embedding_model_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, default="DINOV2-VITB14"
    )
    excluded_classes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    # Geographic location for geofenced models
    country_code: Mapped[str | None] = mapped_column(
        String(3), nullable=True
    )
    state_code: Mapped[str | None] = mapped_column(
        String(2), nullable=True
    )

    # IANA timezone name (e.g. "Europe/Amsterdam"). Stored as metadata
    # and consumed by the future suncalc overlay on the activity
    # pattern chart. Not used to convert any stored datetimes — camera
    # clocks are already in their local timezone.
    timezone: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    # Detection and processing settings
    detection_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5
    )
    event_smoothing: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    smoothing_strength: Mapped[str] = mapped_column(
        String(20), nullable=False, default="normal"
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

    # Verification shortcut labels (keys 1-5 → label options)
    shortcut_labels: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Video processing settings
    video_fps: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0  # Frames per second to extract
    )

    # Clustering defaults (HDBSCAN params)
    min_cluster_size: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5
    )
    min_samples: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3
    )

    # Per-project subprocess batch size overrides.
    # NULL = let the subprocess use its own built-in default (which
    # auto-detects GPU inside its own conda env). A non-null integer
    # is the user's Custom override from the Performance card.
    detection_batch_size: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    classification_batch_size: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    embedding_batch_size: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    # Postprocessing state — SHA-256 hash of last-applied smoothing settings
    postprocessing_settings_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    # Max detections loaded into the verify-tab Observations grid in a
    # single similarity sort. Higher values let large projects render
    # without filters but cost more SQL time, more memory in the
    # subprocess, and a longer wait. Surfaced in Settings → Verification.
    observations_max_detections: Mapped[int] = mapped_column(
        Integer, nullable=False, default=20000
    )

    # Project card thumbnail (absolute path to resized JPEG on disk)
    thumbnail_path: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # Relationships
    sites: Mapped[list["Site"]] = relationship(
        "Site", back_populates="project", cascade="all, delete-orphan"
    )
    deployments: Mapped[list["Deployment"]] = relationship(
        "Deployment", back_populates="project", cascade="all, delete-orphan"
    )
    deployment_queue: Mapped[list["DeploymentQueue"]] = relationship(
        "DeploymentQueue", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name})>"
