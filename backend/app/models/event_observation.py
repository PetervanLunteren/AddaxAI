"""
EventObservation model - per-species MaxN counts within events.

MaxN is the maximum number of individuals of a species visible in any
single image within an event. This prevents double-counting animals
that appear across multiple frames.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .event import Event
    from .file import File
    from .label_taxonomy import LabelTaxonomy


class EventObservation(Base):
    """
    Per-species observation count within an event.

    Each row represents one species (or category like person/vehicle)
    detected in an event, with its MaxN count and the image where
    that peak count was observed.
    """

    __tablename__ = "event_observations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("events.id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    label_taxonomy_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("label_taxonomy.id", ondelete="SET NULL"),
        nullable=True,
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    max_n: Mapped[int] = mapped_column(Integer, nullable=False)
    max_n_file_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("files.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    event: Mapped["Event"] = relationship(
        "Event", back_populates="observations"
    )
    max_n_file: Mapped["File | None"] = relationship("File")
    label_taxonomy: Mapped["LabelTaxonomy | None"] = relationship(
        "LabelTaxonomy", back_populates="observations"
    )

    __table_args__ = (
        UniqueConstraint(
            "event_id", "label_taxonomy_id",
            name="uq_event_obs_event_taxonomy",
        ),
        Index("idx_event_obs_event", "event_id"),
        Index("idx_event_obs_label", "label"),
        Index("idx_event_obs_label_taxonomy", "label_taxonomy_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<EventObservation(event_id={self.event_id}, "
            f"label={self.label}, max_n={self.max_n})>"
        )
