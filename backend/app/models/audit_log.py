"""
Audit log model - track all data changes.

Datetime conventions (see DEVELOPERS.md "Datetime conventions" section):
- `created_at_utc` is a tz-aware UTC audit timestamp.
"""

import uuid
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import JSON, DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

AuditAction = Literal["create", "update", "delete"]


class AuditLog(Base):
    """
    Audit log for tracking all data changes.

    Records who changed what and when, with before/after values.
    Immutable - never updated or deleted.
    """

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'project', 'site', 'detection', etc.
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)  # AuditAction
    user_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )  # Future: for multi-user support
    changes: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True
    )  # Before/after values
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # Indexes
    __table_args__ = (
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_created", "created_at_utc"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, entity_type={self.entity_type}, action={self.action})>"
