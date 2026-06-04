"""require site coords + drop no-coord / null-island sites

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-04-20 17:00:00.000000

Flips the data model so every site has GPS. NULL coords and the buggy
``(0.0, 0.0)`` placeholder both collapsed onto the same semantic state
("I don't have a location") but were hard to distinguish from real
sites. After this migration the only way to represent "no location"
for a deployment is ``site_id IS NULL``.

- Sites with ``latitude IS NULL`` or ``longitude IS NULL`` are
  fingerprinted as no-coord. Their deployments get ``site_id=NULL``;
  the site row is deleted.
- Sites at exactly ``(0.0, 0.0)`` are treated as the same bucket:
  the old Add-Site modal defaulted to 0/0, so these rows are almost
  always the buggy default rather than a real Null Island camera.
- ``sites.latitude`` and ``sites.longitude`` are then altered to
  NOT NULL.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c9d0e1f2a3b4'
down_revision: Union[str, None] = 'b8c9d0e1f2a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Collect sites that either have missing coords or sit at Null
    #    Island (the old form's default). Both buckets become
    #    "no site" on the deployments that referenced them.
    bad_site_ids = [
        row[0]
        for row in conn.execute(
            sa.text(
                "SELECT id FROM sites "
                "WHERE latitude IS NULL OR longitude IS NULL "
                "OR (latitude = 0 AND longitude = 0)"
            )
        ).fetchall()
    ]
    if bad_site_ids:
        placeholders = ",".join(f":id{i}" for i in range(len(bad_site_ids)))
        params = {f"id{i}": sid for i, sid in enumerate(bad_site_ids)}
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

    # 2. Now that every remaining site has real coords, enforce that
    #    at the schema level.
    with op.batch_alter_table("sites") as batch_op:
        batch_op.alter_column(
            "latitude", existing_type=sa.Float(), nullable=False
        )
        batch_op.alter_column(
            "longitude", existing_type=sa.Float(), nullable=False
        )

    # 3. Switch Deployment.site_id FK from CASCADE to SET NULL. Under
    #    the new model, deleting a site shouldn't drag its deployments
    #    down; they just become "no site assigned" which is already a
    #    first-class state.
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.drop_constraint(
            "fk_deployments_site_id", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_deployments_site_id",
            "sites",
            ["site_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    # We can't resurrect the sites we deleted. Reverting the schema
    # only requires relaxing the NOT NULL constraints and restoring
    # the original CASCADE FK. Any deployment currently at
    # site_id=NULL stays that way.
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.drop_constraint(
            "fk_deployments_site_id", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_deployments_site_id",
            "sites",
            ["site_id"],
            ["id"],
            ondelete="CASCADE",
        )

    with op.batch_alter_table("sites") as batch_op:
        batch_op.alter_column(
            "latitude", existing_type=sa.Float(), nullable=True
        )
        batch_op.alter_column(
            "longitude", existing_type=sa.Float(), nullable=True
        )
