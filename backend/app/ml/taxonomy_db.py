"""
Taxonomy DB population — sync taxonomy.csv and rollup entries to species_taxonomy table.

Called during classification (detection_worker) and postprocessing to keep
the species_taxonomy table in sync with what's in Detection.species.
"""

import csv
import json
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models.detection import Detection
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


def populate_taxonomy_from_json(
    model_id: str, json_path: Path, db: Session
) -> int:
    """
    Upsert SpeciesTaxonomy rows from a SpeciesNet results.json file.

    Parses ``classification_category_descriptions`` which contain
    semicolon-delimited strings like:
        UUID;class;order;family;genus;species;common_name

    Uses the **common name** (last field) as ``species_taxonomy.name``
    to match ``Detection.species``.

    Idempotent: skips rows where (model_id, name) already exists.

    Returns:
        Count of newly inserted rows.
    """
    if not json_path.exists():
        logger.warning(f"Results JSON not found: {json_path}")
        return 0

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    descriptions = data.get("classification_category_descriptions", {})
    if not descriptions:
        return 0

    # Query existing names for this model to skip duplicates
    existing = {
        r.name
        for r in db.query(SpeciesTaxonomy.name)
        .filter(SpeciesTaxonomy.classification_model_id == model_id)
        .all()
    }

    inserted = 0
    for desc_str in descriptions.values():
        parts = desc_str.split(";")
        if len(parts) < 7:
            continue

        # Parts: UUID, class, order, family, genus, species, common_name
        taxon_class = parts[1].strip().lower()
        taxon_order = parts[2].strip().lower()
        taxon_family = parts[3].strip().lower()
        taxon_genus = parts[4].strip().lower()
        taxon_species = parts[5].strip().lower()
        common_name = parts[6].strip().lower()

        # Skip entries with no taxonomy (e.g. "blank")
        if not common_name or not any([taxon_class, taxon_order, taxon_family, taxon_genus, taxon_species]):
            continue

        if common_name in existing:
            continue

        taxon = {}
        if taxon_class:
            taxon["class"] = taxon_class
        if taxon_order:
            taxon["order"] = taxon_order
        if taxon_family:
            taxon["family"] = taxon_family
        if taxon_genus:
            taxon["genus"] = taxon_genus
        if taxon_species:
            taxon["species"] = taxon_species

        entry = SpeciesTaxonomy(
            classification_model_id=model_id,
            name=common_name,
            taxon_class=taxon.get("class"),
            taxon_order=taxon.get("order"),
            taxon_family=taxon.get("family"),
            taxon_genus=taxon.get("genus"),
            taxon_species=taxon.get("species"),
            level=_determine_level(taxon),
            is_custom=False,
        )
        db.add(entry)
        existing.add(common_name)
        inserted += 1

    if inserted:
        db.commit()
        logger.info(
            f"Populated {inserted} taxonomy entries from JSON for model {model_id}"
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


BUILTIN_MODEL_ID = "__builtin__"

BUILTIN_LABELS = [
    {"name": "person", "category": "person"},
    {"name": "vehicle", "category": "vehicle"},
]


def ensure_builtin_labels(db: Session) -> int:
    """
    Ensure SpeciesTaxonomy has rows for non-species labels ("person", "vehicle").

    These use classification_model_id="__builtin__" and level="none".
    Idempotent: skips rows that already exist.

    Returns:
        Count of newly inserted rows.
    """
    existing = {
        r.name
        for r in db.query(SpeciesTaxonomy.name)
        .filter(SpeciesTaxonomy.classification_model_id == BUILTIN_MODEL_ID)
        .all()
    }

    inserted = 0
    for label in BUILTIN_LABELS:
        if label["name"] in existing:
            continue
        entry = SpeciesTaxonomy(
            classification_model_id=BUILTIN_MODEL_ID,
            name=label["name"],
            level="none",
            is_custom=False,
        )
        db.add(entry)
        inserted += 1

    if inserted:
        db.commit()
        logger.info(f"Seeded {inserted} builtin label(s) in species_taxonomy")

    return inserted


def link_detections_to_taxonomy(project_id: str, db: Session) -> int:
    """
    Batch-link detections to SpeciesTaxonomy rows via species_taxonomy_id FK.

    For each distinct Detection.species value in the project that has
    species_taxonomy_id IS NULL, finds the matching SpeciesTaxonomy row
    (model-level first, then custom, then builtin) and bulk-updates.

    One UPDATE per species string — not per detection.

    Returns:
        Count of detections linked.
    """
    from app.models import Deployment, File, Project, Site

    # Get the project's classification model
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return 0

    model_id = project.classification_model_id

    # Subquery: file IDs belonging to this project
    project_file_ids = (
        db.query(File.id)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .subquery()
    )

    # Get distinct species names with unlinked detections
    unlinked_species = (
        db.query(func.distinct(Detection.species))
        .filter(
            Detection.file_id.in_(db.query(project_file_ids.c.id)),
            Detection.species.isnot(None),
            Detection.species_taxonomy_id.is_(None),
        )
        .all()
    )
    species_names = [row[0] for row in unlinked_species]

    if not species_names:
        return 0

    # Build lookup: species name → taxonomy ID
    # Priority: model-level > custom > builtin
    name_to_taxonomy_id: dict[str, str] = {}

    # 1. Model-level taxonomy (if model exists)
    if model_id:
        model_rows = (
            db.query(SpeciesTaxonomy.id, SpeciesTaxonomy.name)
            .filter(
                SpeciesTaxonomy.classification_model_id == model_id,
                SpeciesTaxonomy.project_id.is_(None),
                SpeciesTaxonomy.name.in_(species_names),
            )
            .all()
        )
        for tid, name in model_rows:
            name_to_taxonomy_id[name] = tid

    # 2. Custom species for this project
    custom_rows = (
        db.query(SpeciesTaxonomy.id, SpeciesTaxonomy.name)
        .filter(
            SpeciesTaxonomy.project_id == project_id,
            SpeciesTaxonomy.is_custom == True,  # noqa: E712
            SpeciesTaxonomy.name.in_(species_names),
        )
        .all()
    )
    for tid, name in custom_rows:
        if name not in name_to_taxonomy_id:
            name_to_taxonomy_id[name] = tid

    # 3. Builtin labels (person, vehicle)
    builtin_rows = (
        db.query(SpeciesTaxonomy.id, SpeciesTaxonomy.name)
        .filter(
            SpeciesTaxonomy.classification_model_id == BUILTIN_MODEL_ID,
            SpeciesTaxonomy.name.in_(species_names),
        )
        .all()
    )
    for tid, name in builtin_rows:
        if name not in name_to_taxonomy_id:
            name_to_taxonomy_id[name] = tid

    if not name_to_taxonomy_id:
        return 0

    # Bulk-update: one UPDATE per species
    total_linked = 0
    for species_name, taxonomy_id in name_to_taxonomy_id.items():
        count = (
            db.query(Detection)
            .filter(
                Detection.file_id.in_(db.query(project_file_ids.c.id)),
                Detection.species == species_name,
                Detection.species_taxonomy_id.is_(None),
            )
            .update(
                {Detection.species_taxonomy_id: taxonomy_id},
                synchronize_session=False,
            )
        )
        total_linked += count

    if total_linked:
        db.commit()
        logger.info(
            f"Linked {total_linked} detections to taxonomy "
            f"({len(name_to_taxonomy_id)} species) in project {project_id}"
        )

    return total_linked


def resolve_taxonomy_id(species_name: str, project_id: str, db: Session) -> str | None:
    """
    Look up the SpeciesTaxonomy ID for a species name within a project.

    Priority: model-level → custom → builtin. Returns None if no match.
    """
    from app.models import Project

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return None

    model_id = project.classification_model_id

    # 1. Model-level taxonomy
    if model_id:
        row = (
            db.query(SpeciesTaxonomy.id)
            .filter(
                SpeciesTaxonomy.classification_model_id == model_id,
                SpeciesTaxonomy.project_id.is_(None),
                SpeciesTaxonomy.name == species_name,
            )
            .first()
        )
        if row:
            return row[0]

    # 2. Custom species for this project
    row = (
        db.query(SpeciesTaxonomy.id)
        .filter(
            SpeciesTaxonomy.project_id == project_id,
            SpeciesTaxonomy.is_custom == True,  # noqa: E712
            SpeciesTaxonomy.name == species_name,
        )
        .first()
    )
    if row:
        return row[0]

    # 3. Builtin labels
    row = (
        db.query(SpeciesTaxonomy.id)
        .filter(
            SpeciesTaxonomy.classification_model_id == BUILTIN_MODEL_ID,
            SpeciesTaxonomy.name == species_name,
        )
        .first()
    )
    if row:
        return row[0]

    return None
