"""Every detector class gets a builtin label row

A box without a species points at the builtin label row of its detector
class: the label filter tree, "Select all" and the colours match on
``label_taxonomy_id``. Only "animal", "person" and "vehicle" had such a
row, so every box from an underwater detector ("fish", "elasmobranch")
was left with no row and was missing from every label filter.

The app now creates the row for any class on first use. This step does
the same for what is already stored:

1. Every class with an unlabelled, unlinked box, or with a count row
   that has no taxonomy row and carries no species (label empty or the
   class itself), gets a builtin row unless it has one. Name the class,
   both display names the capitalised class, level "none", the same
   shape ``taxonomy_db.ensure_builtin_labels`` writes.
2. Those boxes point at the row and carry its names.
3. Those count rows point at the row and get ``label = category``, the
   shape ``calculate_max_n_for_event`` gives an AI row of a class. Without
   it the key of an existing AI row would change from the text to the id
   at the next MaxN rebuild, and a count a person set on it would come
   back as a second row.

Data only, no schema change. Fills empty pointers and never deletes, so
a database with nothing to fix (any camera trap install) matches no rows
and a second run changes nothing. Downgrade is a no-op: the old state was
the bug. Pinned by ``tests/db/test_migration_data.py``.

Revision ID: a8b2c4d6e8f0
Revises: f769a2f4166a
Create Date: 2026-11-19 12:00:00

"""
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a8b2c4d6e8f0"
down_revision: str | None = "f769a2f4166a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BUILTIN = "__builtin__"


def upgrade() -> None:
    bind = op.get_bind()

    classes = bind.execute(
        sa.text(
            "SELECT DISTINCT category FROM detections "
            "WHERE label IS NULL AND label_taxonomy_id IS NULL "
            "UNION "
            "SELECT DISTINCT category FROM event_observations "
            "WHERE label_taxonomy_id IS NULL "
            "AND (label IS NULL OR label = category)"
        )
    ).scalars().all()

    for category in classes:
        if not category:
            continue
        name = category.capitalize()  # taxonomy_db.builtin_display_name
        tid = bind.execute(
            sa.text(
                "SELECT MIN(id) FROM label_taxonomy "
                "WHERE classification_model_id = :b AND name = :c"
            ),
            {"b": _BUILTIN, "c": category},
        ).scalar()
        if tid is None:
            tid = str(uuid.uuid4())
            bind.execute(
                sa.text(
                    "INSERT INTO label_taxonomy (id, classification_model_id, "
                    "name, level, common_name, scientific_name, is_custom, "
                    "project_id, created_at_utc) VALUES (:id, :b, :c, 'none', "
                    ":n, :n, 0, NULL, :now)"
                ),
                {
                    "id": tid,
                    "b": _BUILTIN,
                    "c": category,
                    "n": name,
                    # The format the ORM writes this column in.
                    "now": datetime.now(UTC).replace(tzinfo=None).isoformat(sep=" "),
                },
            )
        bind.execute(
            sa.text(
                "UPDATE detections SET label_taxonomy_id = :t, "
                "scientific_name = :n, common_name = :n "
                "WHERE label IS NULL AND label_taxonomy_id IS NULL "
                "AND category = :c"
            ),
            {"t": tid, "n": name, "c": category},
        )
        bind.execute(
            sa.text(
                "UPDATE event_observations SET label_taxonomy_id = :t, label = :c "
                "WHERE label_taxonomy_id IS NULL AND category = :c "
                "AND (label IS NULL OR label = :c)"
            ),
            {"t": tid, "c": category},
        )


def downgrade() -> None:
    """No-op: the rows and links stand for data the app now writes itself,
    and the state before was the bug this fixes."""
