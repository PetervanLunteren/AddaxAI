"""
Audit log model - track all data changes.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit action tracking
- Immutable audit trail
"""

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import DateTime, Index, JSON, String
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
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Indexes
    __table_args__ = (
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, entity_type={self.entity_type}, action={self.action})>"
