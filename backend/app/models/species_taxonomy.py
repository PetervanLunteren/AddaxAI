"""
SpeciesTaxonomy model — one row per species/taxon per classification model.

Stores parsed taxonomy data from taxonomy.csv and rolled-up entries from
taxonomic rollup. Enables server-side species filter tree building.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SpeciesTaxonomy(Base):
    """A species or rolled-up taxon entry for a classification model."""

    __tablename__ = "species_taxonomy"

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
    is_custom: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    project_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "classification_model_id", "name", "project_id",
            name="uq_species_taxonomy_model_name_project",
        ),
        Index("idx_species_taxonomy_model", "classification_model_id"),
        Index("idx_species_taxonomy_name", "name"),
        Index("idx_species_taxonomy_project", "project_id"),
    )

    def __repr__(self) -> str:
        return f"<SpeciesTaxonomy {self.name} ({self.level}) model={self.classification_model_id}>"
