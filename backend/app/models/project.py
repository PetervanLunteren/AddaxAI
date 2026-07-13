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

from app.core.confidence import (
    DEFAULT_CLASSIFICATION_GATE,
    DEFAULT_COUNTING_THRESHOLD,
)
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
        String(100), nullable=True, default="DINOV2-VITS14"
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

    # IANA timezone name (e.g. "Europe/Amsterdam"). Consumed by the
    # sun-based insights (sun-band overlay, sun-time transform, diel
    # classification). NOT used to convert any stored datetimes — camera
    # clocks are already in their local timezone.
    #
    # Nullable: a new project starts with NO timezone and auto-derives one
    # from its first site's coordinates (see crud.site.create_site +
    # utils.timezone_from_coords). NULL means "auto / not set yet"; the
    # serialization layer falls back to UTC, and sun features show the
    # existing "needs a site with GPS" banner until coordinates exist.
    timezone: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    # Detection and processing settings.
    #
    # counting_threshold is the counting / visualization filter (what
    # the app shows and counts); classification_gate is the detection
    # confidence above which animal crops are classified and embedded.
    # MegaDetector itself always runs untresholded (0.005, see
    # app/ml/detection); neither setting affects what is stored.
    counting_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=DEFAULT_COUNTING_THRESHOLD
    )
    classification_gate: Mapped[float] = mapped_column(
        Float, nullable=False, default=DEFAULT_CLASSIFICATION_GATE
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
    # The rollup confidence threshold is fixed policy
    # (app.core.confidence.ROLLUP_THRESHOLD), not a per-project setting.
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

    # MegaDetector inference options (advanced, rarely needed). Both are
    # inference-time: they change the raw detector pass, so they apply to
    # future analyses only (like detection_model_id), never retroactively.
    # detection_augment: run the detector on augmented copies and merge
    # (slower, may add false positives). detection_image_size: override the
    # long-edge resize; NULL = MegaDetector's model-native default (1280).
    detection_augment: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    detection_image_size: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    # Postprocessing state — SHA-256 hash of last-applied smoothing settings
    postprocessing_settings_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    # Project card thumbnail (absolute path to resized JPEG on disk)
    thumbnail_path: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # Workflow mode. 'research' is the full Sites/Deployments/Insights
    # workspace; 'folder_run' is the legacy-style point-at-a-folder
    # workflow that lives behind a hidden single deployment. Promotion
    # from 'folder_run' to 'research' just flips this column.
    mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default="research", server_default="research", index=True
    )

    # Stepper state for an in-progress folder run (current step, save
    # options, output dir). NULL for research projects. Cleared when a
    # folder run is promoted.
    folder_run_state: Mapped[dict | None] = mapped_column(
        JSON, nullable=True
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
