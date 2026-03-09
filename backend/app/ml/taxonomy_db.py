"""
Taxonomy DB population — sync taxonomy.csv and rollup entries to species_taxonomy table.

Called during classification (detection_worker) and postprocessing to keep
the species_taxonomy table in sync with what's in Detection.species.
"""

import csv
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models.species_taxonomy import SpeciesTaxonomy

logger = get_logger(__name__)

TAXONOMY_LEVELS = ["class", "order", "family", "genus", "species"]


def _determine_level(taxon: dict[str, str]) -> str:
    """Return the most specific non-empty taxonomy level."""
    for level in reversed(TAXONOMY_LEVELS):
        if taxon.get(level):
            return level
    return "unknown"


def populate_taxonomy_from_csv(
    model_id: str, csv_path: Path, db: Session
) -> int:
    """
    Upsert SpeciesTaxonomy rows from a taxonomy.csv file.

    Idempotent: skips rows where (model_id, name) already exists.

    Returns:
        Count of newly inserted rows.
    """
    if not csv_path.exists():
        logger.warning(f"Taxonomy CSV not found: {csv_path}")
        return 0

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return 0

    # Query existing names for this model to skip duplicates
    existing = {
        r.name
        for r in db.query(SpeciesTaxonomy.name)
        .filter(SpeciesTaxonomy.classification_model_id == model_id)
        .all()
    }

    inserted = 0
    for row in rows:
        name = row.get("model_class", "").strip().lower()
        if not name or name in existing:
            continue

        taxon = {}
        for level in TAXONOMY_LEVELS:
            val = row.get(level, "").strip().lower()
            if val:
                taxon[level] = val

        entry = SpeciesTaxonomy(
            classification_model_id=model_id,
            name=name,
            taxon_class=taxon.get("class"),
            taxon_order=taxon.get("order"),
            taxon_family=taxon.get("family"),
            taxon_genus=taxon.get("genus"),
            taxon_species=taxon.get("species"),
            level=_determine_level(taxon),
            is_custom=False,
        )
        db.add(entry)
        existing.add(name)
        inserted += 1

    if inserted:
        db.commit()
        logger.info(
            f"Populated {inserted} taxonomy entries for model {model_id}"
        )

    return inserted


def add_rollup_taxonomy_entry(
    model_id: str,
    name: str,
    level: str,
    taxonomy_lookup: dict[str, dict[str, str]],
    db: Session,
) -> bool:
    """
    Insert a single rolled-up taxonomy entry (e.g. name="bovidae", level="family").

    Fills ancestor columns from taxonomy_lookup (any entry sharing that taxon value).
    Idempotent: returns False if (model_id, name) already exists.

    Returns:
        True if inserted, False if skipped.
    """
    exists = (
        db.query(SpeciesTaxonomy.id)
        .filter(
            SpeciesTaxonomy.classification_model_id == model_id,
            SpeciesTaxonomy.name == name,
        )
        .first()
    )
    if exists:
        return False

    # Find ancestor columns from any taxonomy entry that has this taxon value
    ancestors: dict[str, str] = {}
    for entry in taxonomy_lookup.values():
        if entry.get(level) == name:
            ancestors = entry
            break

    entry = SpeciesTaxonomy(
        classification_model_id=model_id,
        name=name,
        taxon_class=ancestors.get("class"),
        taxon_order=ancestors.get("order"),
        taxon_family=ancestors.get("family"),
        taxon_genus=ancestors.get("genus") if level in ("genus", "species") else None,
        taxon_species=None,  # Rolled-up entries never have species
        level=level,
        is_custom=False,
    )
    db.add(entry)
    db.commit()

    logger.info(f"Added rollup taxonomy entry: {name} ({level}) for model {model_id}")
    return True
