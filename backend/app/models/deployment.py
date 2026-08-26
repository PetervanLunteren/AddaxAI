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

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .event import Event
    from .file import File
    from .project import Project
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
    # Direct project linkage. Required so a deployment still knows its
    # project when site is unknown (users can run deployment-agnostic
    # batches where site is left blank).
    project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Site is optional: null means the data spans multiple sites or the
    # location is unknown. When a site is deleted its deployments stay
    # (site_id goes NULL) rather than cascading away.
    site_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("sites.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
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

    # The subfolders of this folder are dependent cameras: one animal
    # triggers more than one of them. Events cluster across the
    # subfolders and effort counts once. Off: each subfolder is its own
    # camera or card period. See DEVELOPERS.md "Paired cameras".
    paired_cameras: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0", default=False
    )

    # Audit: the datetime offset (seconds) that was applied to all file
    # timestamps when this deployment was analyzed. Informational only,
    # not used in queries. NULL = no offset was applied.
    datetime_offset_seconds: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    # Audit: the classification gate this deployment was analysed with
    # (the project's value at run time). The project-level setting can
    # change between runs, so mixed-gate projects need the per-run
    # record to explain what was classified / embedded and why.
    # Informational only, not used in queries. NULL = pre-gate analysis.
    classification_gate_used: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    # Non-fatal issues encountered during analysis. List of typed dicts
    # mirroring the queue-entry warnings format: `missing_timestamp`
    # files dropped from ingest, `video_processing_failure` videos that
    # MegaDetector could not decode, etc. Copied from the queue entry
    # on completion so the user can still see what was skipped after
    # the queue row gets cleaned up. NULL when the deployment ran
    # cleanly with nothing to flag.
    warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    project: Mapped["Project"] = relationship(
        "Project", back_populates="deployments"
    )
    site: Mapped["Site | None"] = relationship(
        "Site", back_populates="deployments"
    )
    # passive_deletes=True: the DB owns the cascade (see DEVELOPERS.md,
    # "Deleting analysis data"). Without it, deleting a deployment loads
    # every File and Event into memory first.
    files: Mapped[list["File"]] = relationship(
        "File",
        back_populates="deployment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    events: Mapped[list["Event"]] = relationship(
        "Event",
        back_populates="deployment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<Deployment(id={self.id}, project_id={self.project_id}, "
            f"site_id={self.site_id}, start_date_local={self.start_date_local})>"
        )
