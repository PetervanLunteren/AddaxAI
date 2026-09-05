"""tracks table and detections.track_id

One row per animal a tracker followed through a video (models/track.py),
and a nullable ``track_id`` on ``detections`` pointing at it. Written by
an analysis with ``video_tracking`` on; every existing row stays NULL.

``detections.track_id`` is ``ON DELETE SET NULL``: a track row never goes
on its own outside the purge, and a box must not vanish because its
track did. ``tracks.file_id`` cascades with the file like every other
child of ``files``.

Guarded against drifted beta DBs (DEVELOPERS.md): each step is skipped
when the live schema is already in the target state.

Revision ID: d4e5f6a7b8c0
Revises: c3d4e5f6a7b9
Create Date: 2026-09-05 13:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c0"
down_revision: str | None = "c3d4e5f6a7b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _tables(bind) -> set[str]:
    return set(sa.inspect(bind).get_table_names())


def _columns(bind, table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(bind).get_columns(table)}


def _indexes(bind, table: str) -> set[str]:
    return {i["name"] for i in sa.inspect(bind).get_indexes(table)}


def upgrade() -> None:
    bind = op.get_bind()
    if "tracks" not in _tables(bind):
        op.create_table(
            "tracks",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("file_id", sa.String(length=36), nullable=False),
            sa.Column("track_key", sa.Integer(), nullable=False),
            sa.Column("start_frame", sa.Integer(), nullable=False),
            sa.Column("end_frame", sa.Integer(), nullable=False),
            sa.Column("frame_count", sa.Integer(), nullable=False),
            sa.Column("max_confidence", sa.Float(), nullable=False),
            sa.Column("representative_frame_number", sa.Integer(), nullable=False),
            sa.Column("frame_path", sa.Text(), nullable=True),
            sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("file_id", "track_key", name="uq_tracks_file_key"),
        )
    if "idx_tracks_file" not in _indexes(bind, "tracks"):
        op.create_index("idx_tracks_file", "tracks", ["file_id"], unique=False)

    if "track_id" not in _columns(bind, "detections"):
        # Raw DDL, not op.add_column: alembic refuses a foreign key on an
        # added column in SQLite ("No support for ALTER of constraints")
        # and points at batch mode, which rebuilds the whole detections
        # table (millions of rows on a large install) for one nullable
        # column. SQLite itself accepts a REFERENCES clause on ADD COLUMN
        # when the default is NULL, rewrites the stored CREATE TABLE, and
        # the inspector reads the foreign key back from it, so
        # schema_problems() sees the ON DELETE action like any other.
        op.execute(
            "ALTER TABLE detections ADD COLUMN track_id VARCHAR(36) "
            "REFERENCES tracks(id) ON DELETE SET NULL"
        )
    if "idx_detections_track" not in _indexes(bind, "detections"):
        op.create_index(
            "idx_detections_track", "detections", ["track_id"], unique=False
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "idx_detections_track" in _indexes(bind, "detections"):
        op.drop_index("idx_detections_track", table_name="detections")
    if "track_id" in _columns(bind, "detections"):
        op.execute("ALTER TABLE detections DROP COLUMN track_id")
    if "tracks" in _tables(bind):
        op.drop_table("tracks")
