"""deployment site optional, add deployment.project_id

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-04-20 17:00:00.000000

Makes `deployments.site_id` nullable and changes the FK ondelete from
CASCADE to SET NULL, so users can run deployment-agnostic batches (data
spanning multiple sites, unknown locations, backlogs). Adds a direct
`deployments.project_id` column (NOT NULL, FK to projects.id with
ondelete CASCADE) so project-scoped queries no longer rely on going
through `sites.project_id`. Existing rows are backfilled from the site
relationship before NOT NULL is enforced.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Applies to the batch op below so the pre-existing anonymous FK on
# `deployments.site_id` reflects with a stable name we can drop.
NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Add deployments.project_id as nullable, then backfill from the
    #    related site. Every pre-migration row has a non-null site_id,
    #    so the backfill covers every row. Crash loudly if it doesn't.
    op.add_column(
        "deployments",
        sa.Column("project_id", sa.String(length=36), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE deployments
            SET project_id = (
                SELECT sites.project_id
                FROM sites
                WHERE sites.id = deployments.site_id
            )
            """
        )
    )
    null_rows = bind.execute(
        sa.text("SELECT id FROM deployments WHERE project_id IS NULL")
    ).fetchall()
    if null_rows:
        ids = ", ".join(r[0] for r in null_rows)
        raise RuntimeError(
            f"Cannot backfill deployments.project_id for {len(null_rows)} "
            f"row(s) with no resolvable site: {ids}"
        )

    # 2. Lock down the new column and loosen site_id in a single batch.
    #    The naming_convention makes the pre-existing anonymous site FK
    #    reflect as `fk_deployments_site_id_sites` so we can drop it.
    with op.batch_alter_table(
        "deployments", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.alter_column(
            "project_id", existing_type=sa.String(length=36), nullable=False
        )
        batch_op.create_foreign_key(
            "fk_deployments_project_id_projects",
            "projects",
            ["project_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index(
            "ix_deployments_project_id", ["project_id"], unique=False
        )

        batch_op.alter_column(
            "site_id", existing_type=sa.String(length=36), nullable=True
        )
        batch_op.drop_constraint(
            "fk_deployments_site_id_sites", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_deployments_site_id_sites",
            "sites",
            ["site_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "deployments", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint(
            "fk_deployments_site_id_sites", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_deployments_site_id_sites",
            "sites",
            ["site_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.alter_column(
            "site_id", existing_type=sa.String(length=36), nullable=False
        )

        batch_op.drop_index("ix_deployments_project_id")
        batch_op.drop_constraint(
            "fk_deployments_project_id_projects", type_="foreignkey"
        )

    op.drop_column("deployments", "project_id")
