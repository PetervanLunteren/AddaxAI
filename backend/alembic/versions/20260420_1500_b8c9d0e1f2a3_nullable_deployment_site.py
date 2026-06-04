"""nullable deployment site + direct project fk + drop unknown sites

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-04-20 15:00:00.000000

Restructures the deployment-site relationship so a deployment can
exist without a site (``site_id IS NULL``) and still belong to a
project (``project_id`` is a direct FK). Removes the ``Unknown``
auto-created site placeholder pattern:

- A ``project_id`` column is added to ``deployments``, backfilled from
  the current ``Site.project_id``. Existing code that scoped
  deployments via the Site join can now use the direct FK.
- Sites with ``name='Unknown'`` and NULL coordinates were created as
  placeholders when users left the Site field blank. They're
  fingerprinted and removed: attached deployments go to
  ``site_id=NULL``, then the orphan site row is deleted.
- ``deployments.site_id`` becomes nullable. The existing
  ``ON DELETE CASCADE`` is preserved: deleting a real site still
  cascades its deployments (the expected user intent).
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b8c9d0e1f2a3'
down_revision: Union[str, None] = 'a7b8c9d0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Add deployments.project_id (nullable first so we can backfill).
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.add_column(
            sa.Column("project_id", sa.String(36), nullable=True)
        )

    # 2. Backfill project_id from the current Site.project_id.
    conn.execute(
        sa.text(
            "UPDATE deployments "
            "SET project_id = (SELECT sites.project_id FROM sites "
            "WHERE sites.id = deployments.site_id)"
        )
    )

    # 3. Promote project_id to NOT NULL + FK + index, make site_id nullable.
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.alter_column(
            "project_id",
            existing_type=sa.String(36),
            nullable=False,
        )
        batch_op.alter_column(
            "site_id",
            existing_type=sa.String(36),
            nullable=True,
        )
        batch_op.create_foreign_key(
            "fk_deployments_project_id",
            "projects",
            ["project_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index(
            "ix_deployments_project_id", ["project_id"], unique=False
        )

    # 4. Now that site_id is nullable we can safely detach deployments
    #    from the auto-created Unknown sites before deleting those
    #    site rows. Fingerprint: name='Unknown' with NULL coords.
    unknown_site_ids = [
        row[0]
        for row in conn.execute(
            sa.text(
                "SELECT id FROM sites "
                "WHERE name = 'Unknown' "
                "AND latitude IS NULL AND longitude IS NULL"
            )
        ).fetchall()
    ]
    if unknown_site_ids:
        placeholders = ",".join(f":id{i}" for i in range(len(unknown_site_ids)))
        params = {f"id{i}": sid for i, sid in enumerate(unknown_site_ids)}
        conn.execute(
            sa.text(
                f"UPDATE deployments SET site_id = NULL "
                f"WHERE site_id IN ({placeholders})"
            ),
            params,
        )
        conn.execute(
            sa.text(f"DELETE FROM sites WHERE id IN ({placeholders})"),
            params,
        )


def downgrade() -> None:
    # Not fully reversible: we can't resurrect the deleted Unknown
    # sites. Reverting the schema requires any deployment with
    # site_id=NULL to have been manually reassigned first. If that's
    # the case, the schema changes reverse cleanly.
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.drop_index("ix_deployments_project_id")
        batch_op.drop_constraint("fk_deployments_project_id", type_="foreignkey")
        batch_op.alter_column(
            "site_id",
            existing_type=sa.String(36),
            nullable=False,
        )
        batch_op.drop_column("project_id")
