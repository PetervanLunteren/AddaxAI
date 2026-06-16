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
    func,
)
from sqlalchemy.ext.hybrid import hybrid_property
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
    # AI/box-derived MaxN: the maximum number of this species visible in
    # any single frame within the event.
    max_n: Mapped[int] = mapped_column(Integer, nullable=False)
    max_n_file_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("files.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Human-set count of individuals for this species in the event. When
    # not null it overrides `max_n` for stats and exports (the effective
    # count is `human_count` if set, else `max_n`), letting a verifier
    # record individuals no single frame shows. Mirrors Camtrap-DP /
    # Darwin Core `count` / `organismQuantity`. A human-only row (a
    # species the AI missed) has max_n=0, max_n_file_id=NULL, and
    # human_count set.
    human_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    event: Mapped["Event"] = relationship(
        "Event", back_populates="observations"
    )
    max_n_file: Mapped["File | None"] = relationship("File")
    label_taxonomy: Mapped["LabelTaxonomy | None"] = relationship(
        "LabelTaxonomy", back_populates="observations"
    )

    @hybrid_property
    def effective_count(self) -> int:
        """Human-authoritative count: `human_count` if set, else `max_n`.

        Usable both in Python (`obs.effective_count`) and in SQL
        aggregations (`func.sum(EventObservation.effective_count)`), so
        stats and exports share one definition of "the count".
        """
        return self.human_count if self.human_count is not None else self.max_n

    @effective_count.expression
    def effective_count(cls):  # noqa: N805
        return func.coalesce(cls.human_count, cls.max_n)

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
