"""index files by (deployment_id, file_path) so ingest stops scanning

Ingest looks every file up before inserting it, to decide create-or-update:

    SELECT ... FROM files WHERE file_path = ? AND deployment_id = ?

`file_path` had no index, so the planner used `idx_files_deployment`:

    SEARCH files USING INDEX idx_files_deployment (deployment_id=?)

which is no help at all when a whole library lives in one deployment. It
narrows to the deployment and then compares `file_path` on every row already
inserted, so file N scans N rows and the load is quadratic in the file count.
A beta tester's 1M-image run sat in "Loading to database" for over 12 hours
because of this, at 13% CPU, with the database still slowly growing.

Measured on a synthetic table, one lookup: 1.55 ms at 25k rows, 3.76 ms at 50k,
7.56 ms at 100k, 16.14 ms at 200k. Strictly linear per lookup, so roughly 11
hours of lookups alone for a million files. With this index the same lookup is
5.3 us and flat, and the plan becomes:

    SEARCH files USING INDEX idx_files_deployment_path
        (deployment_id=? AND file_path=?)

`idx_files_deployment` is now subsumed by this one (SQLite can use a leftmost
prefix, including for foreign-key enforcement on the delete cascade) and is
still kept deliberately. DEVELOPERS.md is explicit that indexes existing for
constraint enforcement stay even when they look unused; do not delete it on the
grounds that this index covers it.

Two costs, both small against the win. Inserting a million files carries one
more b-tree, measured at +1 to +2.4 s per million. And deleting a deployment
has one more index to unlink from, roughly +20% on a cascade that DEVELOPERS.md
already flags as slow.

Deliberately NOT unique. `(deployment_id, file_path)` is semantically unique and
the ingest lookup assumes it, but a UNIQUE index that meets a pre-existing
duplicate on someone's database would crash mid-chain and refuse startup, and
there is no clean recovery from that.

Idempotent via if_not_exists, so a half-applied attempt does not loop on
startup.

Revision ID: 7a8b9c0d1e2f
Revises: 6f7a8b9c0d1e
Create Date: 2026-08-04 12:00:00

"""
import logging
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7a8b9c0d1e2f"
down_revision: str | None = "6f7a8b9c0d1e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    # Building this scans the whole files table once. On a large library
    # that is seconds to a minute, during which the backend has not
    # finished starting and the app shows its "Still working" page. Say so
    # in the log, because otherwise the only evidence a support report
    # carries is silence.
    logger.info(
        "Creating idx_files_deployment_path; this scans the files table "
        "once and takes longer the more files you have."
    )
    op.create_index(
        "idx_files_deployment_path",
        "files",
        ["deployment_id", "file_path"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_files_deployment_path",
        table_name="files",
        if_exists=True,
    )
