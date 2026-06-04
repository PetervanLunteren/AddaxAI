"""add deployment name

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-04-20 10:00:00.000000

Adds a human-readable ``name`` field to deployments (required) and the
deployment queue (optional). The add-deployment form lets users type a
name or leave it blank; when blank, the backend fills it from the
folder basename so every Deployment has a scan-friendly identifier.

Existing deployment rows are backfilled in-migration from
``folder_path``. Rows without a folder path (should not exist in
practice) get ``"Deployment <short-id>"`` as a safe fallback.
"""
from __future__ import annotations

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a7b8c9d0e1f2'
down_revision: Union[str, None] = 'f6a7b8c9d0e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _basename_or_fallback(folder_path: str | None, deployment_id: str) -> str:
    """Return folder basename, or a safe fallback if path is empty."""
    if folder_path:
        base = os.path.basename(folder_path.rstrip("/").rstrip("\\"))
        if base:
            return base
    return f"Deployment {deployment_id[:8]}"


def upgrade() -> None:
    # Add name as nullable so we can backfill, then alter to NOT NULL.
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.add_column(sa.Column("name", sa.String(255), nullable=True))

    # Backfill from folder_path basename via the bound connection.
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, folder_path FROM deployments")
    ).fetchall()
    for row in rows:
        name = _basename_or_fallback(row.folder_path, row.id)
        conn.execute(
            sa.text("UPDATE deployments SET name = :name WHERE id = :id"),
            {"name": name, "id": row.id},
        )

    with op.batch_alter_table("deployments") as batch_op:
        batch_op.alter_column(
            "name",
            existing_type=sa.String(255),
            nullable=False,
        )

    # Queue entries carry the user's typed name; NULL means "derive
    # from folder_path at deployment-creation time".
    with op.batch_alter_table("deployment_queue") as batch_op:
        batch_op.add_column(sa.Column("name", sa.String(255), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("deployment_queue") as batch_op:
        batch_op.drop_column("name")

    with op.batch_alter_table("deployments") as batch_op:
        batch_op.drop_column("name")
