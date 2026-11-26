"""
Sequence model - time-clustered groups of files.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `sequence_start_local` / `sequence_end_local` are naive wall-clock times in
  the project's local camera timezone, derived from File.captured_at_local.
- `created_at_utc` is a tz-aware UTC audit timestamp.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import UtcDateTime

if TYPE_CHECKING:
    from .deployment import Deployment
    from .file import File
    from .sequence_observation import SequenceObservation


# Junction table for many-to-many relationship between sequences and files
sequence_files = Table(
    "sequence_files",
    Base.metadata,
    Column(
        "sequence_id",
        String(36),
        ForeignKey("sequences.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("file_id", String(36), ForeignKey("files.id", ondelete="CASCADE"), primary_key=True),
    Column("position", Integer, nullable=True),  # Order within sequence
    # The primary key (sequence_id, file_id) serves lookups by sequence and the
    # cascade from sequences; file_id needs its own.
    Index("idx_sequence_files_file", "file_id"),
)


class Sequence(Base):
    """
    Sequence - time-clustered group of files.

    Sequences are automatically derived from file timestamps.
    Files within a configurable time threshold are grouped into the same sequence.
    """

    __tablename__ = "sequences"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    deployment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("deployments.id", ondelete="CASCADE"), nullable=False
    )
    sequence_start_local: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sequence_end_local: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Human confirmation that the sequence's species and counts are correct
    # (the "Confirm" action on the Observations page). Distinct from
    # Detection.verified ("a label was confirmed", set on the Labels page).
    # Cleared automatically when the sequence's species/count set changes
    # (see crud/sequence_observation).
    confirmed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0", default=False
    )
    # A person's free text about the visit ("camera knocked over", "the
    # juvenile limps"). Never cleared by the app and never lost on a
    # regroup: a regenerated sequence inherits the notes of every old sequence
    # it shares a file with (see crud/sequence, `_snapshot_sequence_carry`).
    # Editing it does not clear `confirmed`. Exported as Camtrap DP
    # `observationComments` on the sequence's rows.
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        UtcDateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    deployment: Mapped["Deployment"] = relationship("Deployment", back_populates="sequences")
    files: Mapped[list["File"]] = relationship(
        "File", secondary=sequence_files, back_populates="sequences"
    )
    # passive_deletes=True: the DB owns the cascade (see DEVELOPERS.md,
    # "Deleting analysis data").
    observations: Mapped[list["SequenceObservation"]] = relationship(
        "SequenceObservation",
        back_populates="sequence",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # Indexes
    __table_args__ = (
        Index("idx_sequences_deployment", "deployment_id"),
        Index("idx_sequences_local", "sequence_start_local", "sequence_end_local"),
    )

    def __repr__(self) -> str:
        return (
            f"<Sequence(id={self.id}, sequence_start_local={self.sequence_start_local}, "
            f"file_count={self.file_count})>"
        )
