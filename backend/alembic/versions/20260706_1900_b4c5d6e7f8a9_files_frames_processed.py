"""files frames_processed

Add a nullable ``frames_processed`` JSON column to ``files``. For
videos it stores the list of analysed frame numbers as written by
MegaDetector's process_video into results.json. The MegaDetector
output format 1.6 requires it (alongside ``frame_rate``) on every
video entry, and the folder-run save step rebuilds the recognition
JSON from the DB, so the list must survive the DB round-trip. NULL
for images and for videos analysed before this column existed; the
recognition JSON omits the field for those until re-analysis.

Guarded against drifted beta DBs (DEVELOPERS.md): add and drop are
each skipped when the live schema is already in the target state.

Revision ID: b4c5d6e7f8a9
Revises: a3b4c5d6e7f8
Create Date: 2026-07-06 19:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4c5d6e7f8a9"
down_revision: str | None = "a3b4c5d6e7f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _files_columns(bind) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns("files")}


def upgrade() -> None:
    bind = op.get_bind()
    if "frames_processed" not in _files_columns(bind):
        op.add_column(
            "files",
            sa.Column("frames_processed", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "frames_processed" in _files_columns(bind):
        op.execute("ALTER TABLE files DROP COLUMN frames_processed")
