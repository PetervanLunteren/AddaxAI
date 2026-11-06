"""
CalibrationAnswer - one answer to "Is this a <label>?" on the Calibration page.

Kept apart from verification on purpose: an answer never writes to the
detection's label or verified state, and verification is never read
here. The AI's label and score are copied in at the moment of answering
and the fit reads those copies, so a reprocess with other settings can
never turn an answer into a wrong one.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CalibrationAnswer(Base):
    __tablename__ = "calibration_answers"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    detection_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("detections.id", ondelete="CASCADE"), nullable=False
    )
    # The AI's label and score when the question was asked.
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    # "right", "wrong" or "unsure" (can't tell; stored, left out of the fit).
    answer: Mapped[str] = mapped_column(String(10), nullable=False)
    answered_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        # One answer per detection, which is also the index the cascade
        # from `detections` needs.
        Index("idx_calibration_answers_detection", "detection_id", unique=True),
        Index("idx_calibration_answers_label", "label"),
    )
