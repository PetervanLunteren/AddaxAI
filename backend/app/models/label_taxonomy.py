"""
LabelTaxonomy model — one row per label/taxon per classification model.

Stores parsed taxonomy data from taxonomy.csv and rolled-up entries from
taxonomic rollup. Enables server-side label filter tree building.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` is a tz-aware UTC audit timestamp.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .detection import Detection
    from .event_observation import EventObservation


class LabelTaxonomy(Base):
    """A label or rolled-up taxon entry for a classification model."""

    __tablename__ = "label_taxonomy"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    classification_model_id: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    name: Mapped[str] = mapped_column(
        String(200), nullable=False
    )
    taxon_class: Mapped[str | None] = mapped_column(String(100), nullable=True)
    taxon_order: Mapped[str | None] = mapped_column(String(100), nullable=True)
    taxon_family: Mapped[str | None] = mapped_column(String(100), nullable=True)
    taxon_genus: Mapped[str | None] = mapped_column(String(100), nullable=True)
    taxon_species: Mapped[str | None] = mapped_column(String(100), nullable=True)
    level: Mapped[str] = mapped_column(
        String(20), nullable=False
    )
    display_name: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    is_custom: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Back-reference to detections linked via FK
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="label_taxonomy"
    )
    observations: Mapped[list["EventObservation"]] = relationship(
        "EventObservation", back_populates="label_taxonomy"
    )

    __table_args__ = (
        UniqueConstraint(
            "classification_model_id", "name", "project_id",
            name="uq_label_taxonomy_model_name_project",
        ),
        Index("idx_label_taxonomy_model", "classification_model_id"),
        Index("idx_label_taxonomy_name", "name"),
        Index("idx_label_taxonomy_project", "project_id"),
    )

    def __repr__(self) -> str:
        return f"<LabelTaxonomy {self.name} ({self.level}) model={self.classification_model_id}>"
