"""Add project_id column to species_taxonomy.

Enables per-project custom species. Each custom species belongs to a specific project.

Revision ID: f7a8b9c0d1e2
Revises: e5f6a7b8c9d1
Create Date: 2026-03-10 12:00:00.000000
"""

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "f7a8b9c0d1e2"
down_revision: Union[str, None] = "e5f6a7b8c9d1"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    with op.batch_alter_table("species_taxonomy") as batch_op:
        batch_op.add_column(
            sa.Column("project_id", sa.String(36), nullable=True)
        )
        batch_op.drop_constraint("uq_species_taxonomy_model_name", type_="unique")
        batch_op.create_unique_constraint(
            "uq_species_taxonomy_model_name_project",
            ["classification_model_id", "name", "project_id"],
        )
        batch_op.create_index("idx_species_taxonomy_project", ["project_id"])


def downgrade() -> None:
    with op.batch_alter_table("species_taxonomy") as batch_op:
        batch_op.drop_index("idx_species_taxonomy_project")
        batch_op.drop_constraint("uq_species_taxonomy_model_name_project", type_="unique")
        batch_op.create_unique_constraint(
            "uq_species_taxonomy_model_name",
            ["classification_model_id", "name"],
        )
        batch_op.drop_column("project_id")
