"""add sequences.camera_key

Which camera of its deployment a sequence comes from: the folder of its
files, or "" for a paired-cameras deployment. The statistics apply the
independence interval per camera with it, so two cameras bundled in one
deployment never join into one independent event.

Existing rows are filled in place, one UPDATE, the same rule as
`services/sequence_clustering.camera_key`: the path of the sequence's first
file up to and including its last separator. `rtrim(path, chars)` strips
every trailing character that is not a separator, so it stops right after
the last `/` or `\\`. A sequence without files gets "".

Idempotent: the column is added only when missing, and the fill can run
again with the same result.

Revision ID: d9f1b3c5e7a1
Revises: c8e0a2b4d6f9
Create Date: 2026-11-26 17:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d9f1b3c5e7a1"
down_revision: str | None = "c8e0a2b4d6f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_column() -> bool:
    return any(
        c["name"] == "camera_key" for c in inspect(op.get_bind()).get_columns("sequences")
    )


def upgrade() -> None:
    if not _has_column():
        op.add_column(
            "sequences",
            sa.Column("camera_key", sa.String(), nullable=False, server_default=""),
        )
    op.execute(
        r"""
        UPDATE sequences SET camera_key = CASE
            WHEN (SELECT paired_cameras FROM deployments d
                  WHERE d.id = sequences.deployment_id) THEN ''
            ELSE coalesce((
                SELECT rtrim(f.file_path,
                             replace(replace(f.file_path, '/', ''), '\', ''))
                FROM sequence_files sf JOIN files f ON f.id = sf.file_id
                WHERE sf.sequence_id = sequences.id
                ORDER BY sf.position, f.file_path
                LIMIT 1
            ), '')
        END
        """
    )


def downgrade() -> None:
    if _has_column():
        op.execute("ALTER TABLE sequences DROP COLUMN camera_key")
