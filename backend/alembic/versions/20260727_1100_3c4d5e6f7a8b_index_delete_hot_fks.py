"""index the two foreign keys that make deletes slow

Both indexes exist purely so SQLite can enforce a foreign key cheaply. Neither
is used by an application query.

`event_observations.max_n_file_id` is the child key of an ON DELETE SET NULL FK
to `files`. With `PRAGMA foreign_keys=ON`, deleting a file makes SQLite look for
rows pointing at it, and with no index that is a full scan of the whole
`event_observations` table, once per deleted file. Deleting a 50k-file run took
8 minutes because of this; with the index it takes seconds.

`detection_embeddings.job_id` is the child key of a NO ACTION FK to `jobs`.
Deleting a project deletes its jobs, and each of those has to prove no embedding
still references it. Same full-scan-per-row shape.

Idempotent via if_not_exists, so a half-applied attempt does not loop on startup.

Revision ID: 3c4d5e6f7a8b
Revises: 2b3c4d5e6f7a
Create Date: 2026-07-27 11:00:00

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c4d5e6f7a8b"
down_revision: str | None = "2b3c4d5e6f7a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "idx_event_obs_max_n_file",
        "event_observations",
        ["max_n_file_id"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_detection_embeddings_job",
        "detection_embeddings",
        ["job_id"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_detection_embeddings_job",
        table_name="detection_embeddings",
        if_exists=True,
    )
    op.drop_index(
        "idx_event_obs_max_n_file",
        table_name="event_observations",
        if_exists=True,
    )
