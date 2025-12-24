"""
Deployment Queue model - queue entries for batch deployment processing.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit relationships
- Clear constraints
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from .project import Project
    from .site import Site


QueueStatus = Literal["pending", "processing", "completed", "failed"]


class DeploymentQueue(Base):
    """
    Queue entry for deployment processing.

    Stores user configuration for creating a deployment and running
    ML models. Processed sequentially when user clicks "Process Queue".
    """

    __tablename__ = "deployment_queue"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    # Step 1: Data
    folder_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Step 2: Deployment
    site_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("sites.id", ondelete="SET NULL"), nullable=True
    )

    # Model configuration now inherited from project (not per-deployment)

    # Processing status
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending, processing, completed, failed
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Result - created deployment ID after processing
    deployment_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )  # FK not enforced to avoid circular dependencies

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="deployment_queue")
    site: Mapped["Site | None"] = relationship("Site")

    def __repr__(self) -> str:
        return f"<DeploymentQueue(id={self.id}, project_id={self.project_id}, status={self.status})>"
