"""drop three indexes that a longer index already covers

Each of these is the leading column of another index on the same table:

- ``idx_files_deployment`` (deployment_id), covered by
  ``idx_files_deployment_path`` (deployment_id, file_path)
- ``idx_detection_embeddings_detection`` (detection_id), covered by
  ``idx_detection_embeddings_detection_model`` (detection_id, embedding_model_id)
- ``idx_event_files_event`` (event_id), covered by the primary key
  (event_id, file_id)

SQLite uses a leftmost prefix for lookups and for foreign-key enforcement,
so the longer index serves both. Checked on SQLite 3.45.1, the build the
app ships, on 2026-11-22: the cascade from deployments, detections and
events seeks the longer index, with no new full scan, and deleting a
27,000-photo project took 2.5 s with and without these three. Each one
cost disk space and an index update on every insert and delete, and
bought nothing.

The migration that added ``idx_files_deployment_path`` (7a8b9c0d1e2f) kept
``idx_files_deployment`` out of caution; that is what this checks.

Revision ID: b9c3d5e7f9a1
Revises: a8b2c4d6e8f0
Create Date: 2026-11-22 12:00:00

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b9c3d5e7f9a1"
down_revision: str | None = "a8b2c4d6e8f0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (index, table, columns)
_DROPPED = (
    ("idx_files_deployment", "files", ["deployment_id"]),
    ("idx_detection_embeddings_detection", "detection_embeddings", ["detection_id"]),
    ("idx_event_files_event", "event_files", ["event_id"]),
)


def upgrade() -> None:
    for name, table, _ in _DROPPED:
        op.drop_index(name, table_name=table, if_exists=True)


def downgrade() -> None:
    for name, table, columns in _DROPPED:
        op.create_index(name, table, columns, if_not_exists=True)
