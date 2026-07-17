"""
Taxonomy DB population — sync taxonomy.csv and rollup entries to label_taxonomy table.

Called during classification (detection_worker) and postprocessing to keep
the label_taxonomy table in sync with what's in Detection.label.
"""

import csv
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.taxonomic_rollup import (
    format_common_name,
    format_scientific_name_from_taxonomy_row,
)
from app.models.detection import Detection
from app.models.label_taxonomy import LabelTaxonomy

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
    Upsert LabelTaxonomy rows from a taxonomy.csv file.

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
        for r in db.query(LabelTaxonomy.name)
        .filter(LabelTaxonomy.classification_model_id == model_id)
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

        entry = LabelTaxonomy(
            classification_model_id=model_id,
            name=name,
            taxon_class=taxon.get("class"),
            taxon_order=taxon.get("order"),
            taxon_family=taxon.get("family"),
            taxon_genus=taxon.get("genus"),
            taxon_species=taxon.get("species"),
            level=_determine_level(taxon),
            common_name=format_common_name(name),
            scientific_name=format_scientific_name_from_taxonomy_row(
                name,
                taxon.get("genus"),
                taxon.get("species"),
                taxon.get("family"),
                taxon.get("order"),
                taxon.get("class"),
            ),
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
        db.query(LabelTaxonomy.id)
        .filter(
            LabelTaxonomy.classification_model_id == model_id,
            LabelTaxonomy.name == name,
        )
        .first()
    )
    if exists:
        return False

    # For species-level rollup the formatted name is "{genus} {species}",
    # but taxonomy_lookup is keyed by the species token alone.
    lookup_value = (
        name.split()[-1] if level == "species" and " " in name else name
    )

    # Find ancestor columns from any taxonomy entry that has this taxon value
    ancestors: dict[str, str] = {}
    for entry in taxonomy_lookup.values():
        if entry.get(level) == lookup_value:
            ancestors = entry
            break

    # Only ranks at or above the rollup level apply. The matched source
    # entry carries the full chain down to species (e.g. for a class
    # rollup the source could be "marmot,mammalia,rodentia,sciuridae,
    # marmota,marmota"); copying the lower ranks would falsely claim the
    # rolled-up taxon belongs to that specific descendant chain. A class
    # rollup is just the class — order/family/genus/species stay NULL.
    rank_order = ["class", "order", "family", "genus", "species"]
    rollup_idx = rank_order.index(level) if level in rank_order else -1

    def ancestor_at(rank: str) -> str | None:
        if rollup_idx < 0:
            return None
        if rank_order.index(rank) > rollup_idx:
            return None
        return ancestors.get(rank)

    species_val = lookup_value if level == "species" else None
    genus_val = ancestor_at("genus")

    if level == "kingdom":
        # Kingdom rollup lands on the literal string "animal", but the
        # scientifically correct kingdom name is "Animalia". Matches the
        # Latin naming we use for other rollup rows (Bovidae, Felidae, ...)
        # and makes the matrix cell stand apart from the detector category.
        scientific_name = "Animalia"
    elif level == "species":
        scientific_name = format_scientific_name_from_taxonomy_row(
            name,
            genus_val,
            species_val,
            ancestor_at("family"),
            ancestor_at("order"),
            ancestor_at("class"),
        )
    else:
        scientific_name = name.capitalize()

    taxonomy_entry = LabelTaxonomy(
        classification_model_id=model_id,
        name=name,
        taxon_class=ancestor_at("class"),
        taxon_order=ancestor_at("order"),
        taxon_family=ancestor_at("family"),
        taxon_genus=genus_val,
        taxon_species=species_val,
        level=level,
        common_name=format_common_name(name),
        scientific_name=scientific_name,
        is_custom=False,
    )
    db.add(taxonomy_entry)
    db.commit()

    logger.info(f"Added rollup taxonomy entry: {name} ({level}) for model {model_id}")
    return True


BUILTIN_MODEL_ID = "__builtin__"

BUILTIN_LABELS = [
    {"name": "animal", "category": "animal"},
    {"name": "person", "category": "person"},
    {"name": "vehicle", "category": "vehicle"},
]


def ensure_builtin_labels(db: Session) -> dict[str, str]:
    """
    Ensure LabelTaxonomy has rows for non-classification labels
    ("animal", "person", "vehicle").

    These use classification_model_id="__builtin__" and level="none".
    Idempotent: skips rows that already exist.

    Returns:
        Mapping of builtin name to taxonomy UUID,
        e.g. {"animal": "uuid-1", "person": "uuid-2", "vehicle": "uuid-3"}.
    """
    existing = {
        r.name: r.id
        for r in db.query(LabelTaxonomy.name, LabelTaxonomy.id)
        .filter(LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID)
        .all()
    }

    inserted = 0
    for label_def in BUILTIN_LABELS:
        if label_def["name"] in existing:
            continue
        taxonomy_entry = LabelTaxonomy(
            classification_model_id=BUILTIN_MODEL_ID,
            name=label_def["name"],
            level="none",
            common_name=label_def["name"].capitalize(),
            scientific_name=label_def["name"].capitalize(),
            is_custom=False,
        )
        db.add(taxonomy_entry)
        db.flush()
        existing[label_def["name"]] = taxonomy_entry.id
        inserted += 1

    if inserted:
        db.commit()
        logger.info(f"Seeded {inserted} builtin label(s) in label_taxonomy")

    return existing


def batch_resolve_taxonomy_ids(
    label_names: list[str],
    model_id: str | None,
    project_id: str,
    db: Session,
) -> dict[str, tuple[str, str | None, str | None]]:
    """
    Resolve a batch of label names to
    (taxonomy_id, scientific_name, common_name) tuples.

    Priority: model-level > custom > builtin.
    Single query per priority level instead of N+1.

    Matching is case-insensitive. `LabelTaxonomy.name` is always written
    lowercase (see populate_taxonomy_from_csv), but `Detection.label`
    keeps whatever case the model emitted, and callers look the result up
    by `label.lower()`. Comparing raw names against the column therefore
    matched nothing for any model whose classes are not already lowercase
    — QLD-WOB-v1 emits `Alectura_lathami` — and the model silently lost
    its whole label tree and its rollup.

    Returns:
        {lowercase_name: (taxonomy_id, scientific_name, common_name)}
        for all matched names.
    """
    if not label_names:
        return {}

    lookup_names = [name.lower() for name in label_names]
    result: dict[str, tuple[str, str | None, str | None]] = {}

    # 1. Model-level taxonomy
    if model_id:
        model_rows = (
            db.query(
                LabelTaxonomy.id,
                LabelTaxonomy.name,
                LabelTaxonomy.scientific_name,
                LabelTaxonomy.common_name,
            )
            .filter(
                LabelTaxonomy.classification_model_id == model_id,
                LabelTaxonomy.project_id.is_(None),
                LabelTaxonomy.name.in_(lookup_names),
            )
            .all()
        )
        for tid, name, sci, common in model_rows:
            result[name.lower()] = (tid, sci, common)

    # 2. Custom labels for this project
    custom_rows = (
        db.query(
            LabelTaxonomy.id,
            LabelTaxonomy.name,
            LabelTaxonomy.scientific_name,
            LabelTaxonomy.common_name,
        )
        .filter(
            LabelTaxonomy.project_id == project_id,
            LabelTaxonomy.is_custom == True,  # noqa: E712
            LabelTaxonomy.name.in_(lookup_names),
        )
        .all()
    )
    for tid, name, sci, common in custom_rows:
        key = name.lower()
        if key not in result:
            result[key] = (tid, sci, common)

    # 3. Builtin labels (animal, person, vehicle)
    builtin_rows = (
        db.query(
            LabelTaxonomy.id,
            LabelTaxonomy.name,
            LabelTaxonomy.scientific_name,
            LabelTaxonomy.common_name,
        )
        .filter(
            LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID,
            LabelTaxonomy.name.in_(lookup_names),
        )
        .all()
    )
    for tid, name, sci, common in builtin_rows:
        key = name.lower()
        if key not in result:
            result[key] = (tid, sci, common)

    return result


def link_detections_to_taxonomy(project_id: str, db: Session) -> int:
    """
    Batch-link detections to LabelTaxonomy rows via label_taxonomy_id FK.

    For each distinct Detection.label value in the project that has
    label_taxonomy_id IS NULL, finds the matching LabelTaxonomy row
    (model-level first, then custom, then builtin) and bulk-updates.

    One UPDATE per label string -- not per detection.

    Returns:
        Count of detections linked.
    """
    from app.models import Deployment, File, Project

    # Get the project's classification model
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return 0

    model_id = project.classification_model_id

    # Subquery: file IDs belonging to this project
    project_file_ids = (
        db.query(File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .filter(Deployment.project_id == project_id)
        .subquery()
    )

    total_linked = 0

    # --- Pass 1: link by Detection.label (classified detections) ---
    unlinked_labels = (
        db.query(func.distinct(Detection.label))
        .filter(
            Detection.file_id.in_(db.query(project_file_ids.c.id)),
            Detection.label.isnot(None),
            Detection.label_taxonomy_id.is_(None),
        )
        .all()
    )
    label_names = [row[0] for row in unlinked_labels]

    # LabelTaxonomy.name is always stored lowercase, but Detection.label
    # keeps whatever case the model emitted (QLD-WOB-v1 emits
    # "Alectura_lathami"). Match on lowercase, and keep the original
    # spellings so the UPDATE below can still find the rows.
    labels_by_lower: dict[str, list[str]] = {}
    for original in label_names:
        labels_by_lower.setdefault(original.lower(), []).append(original)
    lookup_names = list(labels_by_lower)

    if label_names:
        # Build lookup: label name -> (taxonomy_id, scientific_name, common_name)
        # Priority: model-level > custom > builtin
        name_to_taxonomy: dict[str, tuple[str, str | None, str | None]] = {}

        # 1. Model-level taxonomy (if model exists)
        if model_id:
            model_rows = (
                db.query(
                    LabelTaxonomy.id,
                    LabelTaxonomy.name,
                    LabelTaxonomy.scientific_name,
                    LabelTaxonomy.common_name,
                )
                .filter(
                    LabelTaxonomy.classification_model_id == model_id,
                    LabelTaxonomy.project_id.is_(None),
                    LabelTaxonomy.name.in_(lookup_names),
                )
                .all()
            )
            for tid, name, sci, common in model_rows:
                name_to_taxonomy[name] = (tid, sci, common)

        # 2. Custom labels for this project
        custom_rows = (
            db.query(
                LabelTaxonomy.id,
                LabelTaxonomy.name,
                LabelTaxonomy.scientific_name,
                LabelTaxonomy.common_name,
            )
            .filter(
                LabelTaxonomy.project_id == project_id,
                LabelTaxonomy.is_custom == True,  # noqa: E712
                LabelTaxonomy.name.in_(lookup_names),
            )
            .all()
        )
        for tid, name, sci, common in custom_rows:
            if name not in name_to_taxonomy:
                name_to_taxonomy[name] = (tid, sci, common)

        # 3. Builtin labels (animal, person, vehicle)
        builtin_rows = (
            db.query(
                LabelTaxonomy.id,
                LabelTaxonomy.name,
                LabelTaxonomy.scientific_name,
                LabelTaxonomy.common_name,
            )
            .filter(
                LabelTaxonomy.classification_model_id
                == BUILTIN_MODEL_ID,
                LabelTaxonomy.name.in_(lookup_names),
            )
            .all()
        )
        for tid, name, sci, common in builtin_rows:
            if name not in name_to_taxonomy:
                name_to_taxonomy[name] = (tid, sci, common)

        # Bulk-update: one UPDATE per label (set FK + both names).
        # name_to_taxonomy is keyed by the taxonomy's lowercase name, so
        # match against the original Detection.label spellings it came from.
        for label_name, (taxonomy_id, sci, common) in name_to_taxonomy.items():
            originals = labels_by_lower.get(label_name.lower())
            if not originals:
                continue
            count = (
                db.query(Detection)
                .filter(
                    Detection.file_id.in_(
                        db.query(project_file_ids.c.id)
                    ),
                    Detection.label.in_(originals),
                    Detection.label_taxonomy_id.is_(None),
                )
                .update(
                    {
                        Detection.label_taxonomy_id: taxonomy_id,
                        Detection.scientific_name: sci,
                        Detection.common_name: common,
                    },
                    synchronize_session=False,
                )
            )
            total_linked += count

    # --- Pass 2: link by Detection.category (detection-only) ---
    # For detections with label=NULL, match category against builtins.
    unlinked_categories = (
        db.query(func.distinct(Detection.category))
        .filter(
            Detection.file_id.in_(db.query(project_file_ids.c.id)),
            Detection.label.is_(None),
            Detection.label_taxonomy_id.is_(None),
        )
        .all()
    )
    category_names = [row[0] for row in unlinked_categories if row[0]]

    if category_names:
        builtin_cat_rows = (
            db.query(
                LabelTaxonomy.id,
                LabelTaxonomy.name,
                LabelTaxonomy.scientific_name,
                LabelTaxonomy.common_name,
            )
            .filter(
                LabelTaxonomy.classification_model_id
                == BUILTIN_MODEL_ID,
                LabelTaxonomy.name.in_(category_names),
            )
            .all()
        )
        for tid, cat_name, sci, common in builtin_cat_rows:
            count = (
                db.query(Detection)
                .filter(
                    Detection.file_id.in_(
                        db.query(project_file_ids.c.id)
                    ),
                    Detection.label.is_(None),
                    Detection.category == cat_name,
                    Detection.label_taxonomy_id.is_(None),
                )
                .update(
                    {
                        Detection.label_taxonomy_id: tid,
                        Detection.scientific_name: sci,
                        Detection.common_name: common,
                    },
                    synchronize_session=False,
                )
            )
            total_linked += count

    if total_linked:
        db.commit()
        logger.info(
            f"Linked {total_linked} detections to taxonomy "
            f"in project {project_id}"
        )

    return total_linked


def resolve_taxonomy_id(label_name: str, project_id: str, db: Session) -> str | None:
    """
    Look up the LabelTaxonomy ID for a label name within a project.

    Priority: model-level -> custom -> builtin. Returns None if no match.
    """
    from app.models import Project

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return None

    model_id = project.classification_model_id

    # 1. Model-level taxonomy
    if model_id:
        row = (
            db.query(LabelTaxonomy.id)
            .filter(
                LabelTaxonomy.classification_model_id == model_id,
                LabelTaxonomy.project_id.is_(None),
                LabelTaxonomy.name == label_name,
            )
            .first()
        )
        if row:
            return row[0]

    # 2. Custom labels for this project
    row = (
        db.query(LabelTaxonomy.id)
        .filter(
            LabelTaxonomy.project_id == project_id,
            LabelTaxonomy.is_custom == True,  # noqa: E712
            LabelTaxonomy.name == label_name,
        )
        .first()
    )
    if row:
        return row[0]

    # 3. Builtin labels
    row = (
        db.query(LabelTaxonomy.id)
        .filter(
            LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID,
            LabelTaxonomy.name == label_name,
        )
        .first()
    )
    if row:
        return row[0]

    return None
