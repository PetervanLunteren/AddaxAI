"""unify projects: drop kind, relax site coords, require timestamps

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-04-17 17:00:00.000000

Collapses the separate Batch run mode into the regular Projects flow:

- ``projects.kind`` is dropped. Every project is a project; the Batch
  run / Project split was a UI concept we're removing.
- ``sites.latitude`` / ``sites.longitude`` become nullable. NULL means
  "unknown or mixed location". Downstream map and Camtrap-DP features
  disable themselves with an inline nudge when coords are missing.
  Replaces the old practice of inventing (0, 0) for quick runs.
- ``files.captured_at_local`` goes back to NOT NULL. Timestamps remain
  required for every project: event clustering, activity plots,
  smoothing, and independence intervals all depend on them. Missing
  EXIF is once again a hard stop with ``MissingTimestampError``.
- ``deployments.start_date_local`` goes back to NOT NULL. Derived from
  file timestamps in Phase 6.

Aborts if the DB contains any NULL ``captured_at_local`` or
``start_date_local`` rows left over from the short-lived soft-EXIF
experiment; the user should delete those rows manually and retry.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Pre-flight: refuse to re-impose NOT NULL if any existing rows
    # carry NULLs from a previous batch-run experiment. Surface a clear
    # message instead of letting SQLAlchemy fail obscurely on copy.
    null_files = conn.execute(
        sa.text("SELECT COUNT(*) FROM files WHERE captured_at_local IS NULL")
    ).scalar_one()
    if null_files:
        raise RuntimeError(
            f"Cannot upgrade: {null_files} files(s) have NULL captured_at_local. "
            "Delete those rows (they came from the short-lived batch-run path) "
            "and retry."
        )
    null_deployments = conn.execute(
        sa.text("SELECT COUNT(*) FROM deployments WHERE start_date_local IS NULL")
    ).scalar_one()
    if null_deployments:
        raise RuntimeError(
            f"Cannot upgrade: {null_deployments} deployment(s) have NULL "
            "start_date_local. Delete those rows and retry."
        )

    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("kind")

    with op.batch_alter_table("files") as batch_op:
        batch_op.alter_column(
            "captured_at_local",
            existing_type=sa.DateTime(),
            nullable=False,
        )

    with op.batch_alter_table("deployments") as batch_op:
        batch_op.alter_column(
            "start_date_local",
            existing_type=sa.Date(),
            nullable=False,
        )

    with op.batch_alter_table("sites") as batch_op:
        batch_op.alter_column(
            "latitude",
            existing_type=sa.Float(),
            nullable=True,
        )
        batch_op.alter_column(
            "longitude",
            existing_type=sa.Float(),
            nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("sites") as batch_op:
        batch_op.alter_column(
            "longitude",
            existing_type=sa.Float(),
            nullable=False,
        )
        batch_op.alter_column(
            "latitude",
            existing_type=sa.Float(),
            nullable=False,
        )

    with op.batch_alter_table("deployments") as batch_op:
        batch_op.alter_column(
            "start_date_local",
            existing_type=sa.Date(),
            nullable=True,
        )

    with op.batch_alter_table("files") as batch_op:
        batch_op.alter_column(
            "captured_at_local",
            existing_type=sa.DateTime(),
            nullable=True,
        )

    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(
            sa.Column(
                "kind",
                sa.String(20),
                nullable=False,
                server_default="project",
            )
        )
        batch_op.alter_column("kind", server_default=None)
