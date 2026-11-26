"""rename events to sequences

The stored group of files a person confirms counts on is a sequence from
now on; "event" is kept for the computed unit the statistics use. This
renames the three tables, their columns and indexes, and the project's
smoothing flag. Names only: no row is added, removed or changed.

    events              -> sequences
    event_files         -> sequence_files
    event_observations  -> sequence_observations
    *.event_id          -> sequence_id
    event_start_local   -> sequence_start_local
    event_end_local     -> sequence_end_local
    sequence_number     -> position (the file's place in its sequence)
    projects.event_smoothing -> sequence_smoothing

Plain ALTER TABLE, no batch rebuild: SQLite rewrites the foreign keys that
point at a renamed table (since 3.26, unless legacy_alter_table is on, which
this refuses) and carries renamed columns into their indexes. Index names
cannot be renamed, so the seven indexes are dropped and created again under
their new names.

Idempotent: every step checks the old name exists and the new one does not,
so a half-applied attempt finishes instead of failing.

Revision ID: b7d9f1a3c5e7
Revises: fa19b0f2a459
Create Date: 2026-11-26 15:00:00

"""
from collections.abc import Sequence

from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d9f1a3c5e7"
down_revision: str | None = "fa19b0f2a459"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = [
    ("events", "sequences"),
    ("event_files", "sequence_files"),
    ("event_observations", "sequence_observations"),
]

# (table after the table rename, old column, new column)
_COLUMNS = [
    ("sequences", "event_start_local", "sequence_start_local"),
    ("sequences", "event_end_local", "sequence_end_local"),
    ("sequence_files", "event_id", "sequence_id"),
    ("sequence_files", "sequence_number", "position"),
    ("sequence_observations", "event_id", "sequence_id"),
    ("projects", "event_smoothing", "sequence_smoothing"),
]

# The indexes before and after, as (name, table, columns).
_OLD_INDEXES = [
    ("idx_events_deployment", "events", "deployment_id"),
    ("idx_events_local", "events", "event_start_local, event_end_local"),
    ("idx_event_files_file", "event_files", "file_id"),
    ("idx_event_obs_event", "event_observations", "event_id"),
    ("idx_event_obs_label", "event_observations", "label"),
    ("idx_event_obs_label_taxonomy", "event_observations", "label_taxonomy_id"),
    ("idx_event_obs_max_n_file", "event_observations", "max_n_file_id"),
]
_NEW_INDEXES = [
    ("idx_sequences_deployment", "sequences", "deployment_id"),
    ("idx_sequences_local", "sequences", "sequence_start_local, sequence_end_local"),
    ("idx_sequence_files_file", "sequence_files", "file_id"),
    ("idx_sequence_obs_sequence", "sequence_observations", "sequence_id"),
    ("idx_sequence_obs_label", "sequence_observations", "label"),
    ("idx_sequence_obs_label_taxonomy", "sequence_observations", "label_taxonomy_id"),
    ("idx_sequence_obs_max_n_file", "sequence_observations", "max_n_file_id"),
]


def _columns(table: str) -> set[str]:
    return {c["name"] for c in inspect(op.get_bind()).get_columns(table)}


def _rename(tables, columns, drop_indexes, create_indexes) -> None:
    bind = op.get_bind()
    for name, _table, _cols in drop_indexes:
        op.execute(f"DROP INDEX IF EXISTS {name}")
    for old, new in tables:
        present = set(inspect(bind).get_table_names())
        if old in present and new not in present:
            op.execute(f"ALTER TABLE {old} RENAME TO {new}")
    for table, old, new in columns:
        cols = _columns(table)
        if old in cols and new not in cols:
            op.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
    for name, table, cols in create_indexes:
        op.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols})")


def upgrade() -> None:
    if op.get_bind().exec_driver_sql("PRAGMA legacy_alter_table").scalar():
        raise RuntimeError(
            "legacy_alter_table is on, so renaming a table would leave the "
            "foreign keys that point at it on the old name."
        )
    _rename(_TABLES, _COLUMNS, _OLD_INDEXES, _NEW_INDEXES)


def downgrade() -> None:
    old_table = {new: old for old, new in _TABLES}
    _rename(
        [(new, old) for old, new in reversed(_TABLES)],
        [(old_table.get(t, t), new, old) for t, old, new in reversed(_COLUMNS)],
        _NEW_INDEXES,
        _OLD_INDEXES,
    )
