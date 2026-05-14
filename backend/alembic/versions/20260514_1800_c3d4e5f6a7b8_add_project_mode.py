"""add project mode and folder_run_state

Introduces two columns on the `projects` table to support the new
folder-run flow:

- `mode` distinguishes `'research'` projects (the existing app) from
  `'folder_run'` projects (the new legacy-style point-at-a-folder
  workflow). Existing rows backfill to `'research'`.
- `folder_run_state` is a JSON blob carrying the in-progress stepper
  state for a folder run (current step, save options, output dir). It
  is NULL for research projects.

This migration is idempotent: it skips the ADD COLUMN / CREATE INDEX
steps if the column or index already exists. Needed because a previous
half-applied attempt (column added, alembic_version not updated) would
otherwise loop on every startup with a `duplicate column name` error.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-05-14 18:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _projects_has_column(name: str) -> bool:
    inspector = inspect(op.get_bind())
    return any(c["name"] == name for c in inspector.get_columns("projects"))


def _projects_has_index(name: str) -> bool:
    inspector = inspect(op.get_bind())
    return any(idx["name"] == name for idx in inspector.get_indexes("projects"))


def upgrade() -> None:
    if not _projects_has_column("mode"):
        # server_default backfills existing rows to 'research'. Future rows
        # take their default from the SQLAlchemy model, so the server
        # default is only load-bearing during this migration.
        op.add_column(
            "projects",
            sa.Column(
                "mode",
                sa.String(length=16),
                nullable=False,
                server_default="research",
            ),
        )
    if not _projects_has_column("folder_run_state"):
        op.add_column(
            "projects",
            sa.Column(
                "folder_run_state",
                sa.JSON(),
                nullable=True,
            ),
        )
    if not _projects_has_index("ix_projects_mode"):
        op.create_index("ix_projects_mode", "projects", ["mode"])


def downgrade() -> None:
    if _projects_has_index("ix_projects_mode"):
        op.drop_index("ix_projects_mode", table_name="projects")
    if _projects_has_column("folder_run_state"):
        op.drop_column("projects", "folder_run_state")
    if _projects_has_column("mode"):
        op.drop_column("projects", "mode")
