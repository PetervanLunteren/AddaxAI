"""Rename 'species' to 'label' across the database schema.

- Rename table: species_taxonomy -> label_taxonomy
- Rename columns on detections: species -> label,
  species_confidence -> label_confidence,
  species_taxonomy_id -> label_taxonomy_id
- Recreate indexes with updated names
- Migrate JSON blobs in projects.shortcut_labels:
  change inner "species" key to "label"

Revision ID: a1b2c3d4e5f6
Revises: f7a8b9c0d1e2
Create Date: 2026-03-11 10:00:00.000000
"""

import json
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f7a8b9c0d1e2"
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


# ---------------------------------------------------------------------------
# Helpers for the JSON key migration in projects.shortcut_labels
# ---------------------------------------------------------------------------

def _rename_json_key(raw_value, old_key: str, new_key: str):
    """Parse a shortcut_labels JSON blob and rename a key inside each entry."""
    if not raw_value:
        return raw_value

    data = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    if not isinstance(data, dict):
        return raw_value

    changed = False
    for slot_key, entry in data.items():
        if isinstance(entry, dict) and old_key in entry:
            entry[new_key] = entry.pop(old_key)
            changed = True

    return json.dumps(data) if changed else raw_value


def _migrate_shortcut_labels(old_key: str, new_key: str) -> None:
    """Update the inner JSON key in every projects.shortcut_labels row."""
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, shortcut_labels FROM projects WHERE shortcut_labels IS NOT NULL")
    ).fetchall()

    for row_id, raw_value in rows:
        updated = _rename_json_key(raw_value, old_key, new_key)
        if updated != raw_value:
            conn.execute(
                sa.text("UPDATE projects SET shortcut_labels = :val WHERE id = :id"),
                {"val": updated, "id": row_id},
            )


# ---------------------------------------------------------------------------
# Upgrade
# ---------------------------------------------------------------------------

def upgrade() -> None:
    # 1. Rename table: species_taxonomy -> label_taxonomy
    op.rename_table("species_taxonomy", "label_taxonomy")

    # 2. Recreate label_taxonomy indexes with new names.
    #    SQLite cannot rename indexes, so drop old ones and create new ones.
    #    The table name argument must match the *new* table name after rename.
    op.drop_index("idx_species_taxonomy_model", table_name="label_taxonomy")
    op.drop_index("idx_species_taxonomy_name", table_name="label_taxonomy")
    op.drop_index("idx_species_taxonomy_project", table_name="label_taxonomy")
    op.create_index("idx_label_taxonomy_model", "label_taxonomy", ["classification_model_id"])
    op.create_index("idx_label_taxonomy_name", "label_taxonomy", ["name"])
    op.create_index("idx_label_taxonomy_project", "label_taxonomy", ["project_id"])

    # 3. Rename unique constraint on label_taxonomy via batch (SQLite compat).
    with op.batch_alter_table("label_taxonomy") as batch_op:
        batch_op.drop_constraint("uq_species_taxonomy_model_name_project", type_="unique")
        batch_op.create_unique_constraint(
            "uq_label_taxonomy_model_name_project",
            ["classification_model_id", "name", "project_id"],
        )

    # 4. Rename columns on detections and update FK + indexes.
    #    batch_alter_table recreates the table under the hood on SQLite,
    #    so column renames and FK changes are applied atomically.
    #
    #    Note: species_taxonomy_id was never added by a prior migration,
    #    so we ADD label_taxonomy_id instead of renaming it.
    with op.batch_alter_table("detections") as batch_op:
        # Rename columns
        batch_op.alter_column("species", new_column_name="label")
        batch_op.alter_column("species_confidence", new_column_name="label_confidence")

        # Add label_taxonomy_id (no prior column to rename)
        batch_op.add_column(
            sa.Column("label_taxonomy_id", sa.String(36), nullable=True)
        )

        # Drop old indexes
        batch_op.drop_index("idx_detections_species")
        batch_op.drop_index("idx_detections_species_confidence")

        # Create new indexes with updated names
        batch_op.create_index("idx_detections_label", ["label"])
        batch_op.create_index("idx_detections_label_confidence", ["label_confidence"])
        batch_op.create_index("idx_detections_label_taxonomy", ["label_taxonomy_id"])

        # Create FK pointing to the renamed table
        batch_op.create_foreign_key(
            "fk_detections_label_taxonomy_id",
            "label_taxonomy",
            ["label_taxonomy_id"],
            ["id"],
            ondelete="SET NULL",
        )

    # 5. Migrate JSON blobs: "species" -> "label" inside shortcut_labels entries
    _migrate_shortcut_labels(old_key="species", new_key="label")


# ---------------------------------------------------------------------------
# Downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    # 5. Revert JSON blobs: "label" -> "species"
    _migrate_shortcut_labels(old_key="label", new_key="species")

    # 4. Revert column renames on detections
    with op.batch_alter_table("detections") as batch_op:
        batch_op.drop_index("idx_detections_label")
        batch_op.drop_index("idx_detections_label_confidence")
        batch_op.drop_index("idx_detections_label_taxonomy")

        batch_op.alter_column("label", new_column_name="species")
        batch_op.alter_column("label_confidence", new_column_name="species_confidence")

        # Drop the column (it was added in upgrade, not renamed)
        batch_op.drop_column("label_taxonomy_id")

        batch_op.create_index("idx_detections_species", ["species"])
        batch_op.create_index("idx_detections_species_confidence", ["species_confidence"])

    # 3. Revert unique constraint name on the taxonomy table
    with op.batch_alter_table("label_taxonomy") as batch_op:
        batch_op.drop_constraint("uq_label_taxonomy_model_name_project", type_="unique")
        batch_op.create_unique_constraint(
            "uq_species_taxonomy_model_name_project",
            ["classification_model_id", "name", "project_id"],
        )

    # 2. Revert indexes on taxonomy table
    op.drop_index("idx_label_taxonomy_model", table_name="label_taxonomy")
    op.drop_index("idx_label_taxonomy_name", table_name="label_taxonomy")
    op.drop_index("idx_label_taxonomy_project", table_name="label_taxonomy")
    op.create_index("idx_species_taxonomy_model", "species_taxonomy", ["classification_model_id"])
    op.create_index("idx_species_taxonomy_name", "species_taxonomy", ["name"])
    op.create_index("idx_species_taxonomy_project", "species_taxonomy", ["project_id"])

    # 1. Rename table back
    op.rename_table("label_taxonomy", "species_taxonomy")
