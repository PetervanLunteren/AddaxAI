"""
CRUD operations for Detection model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""


from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.detection import (
    DetectionCreate,
    DetectionCreateHuman,
    DetectionUpdate,
)
from app.models import Deployment, Detection, File


def get_detection(db: Session, detection_id: str) -> Detection | None:
    """
    Get detection by ID.

    Returns None if detection doesn't exist.
    """
    result = db.execute(select(Detection).where(Detection.id == detection_id))
    return result.scalar_one_or_none()


def get_detections_by_file(
    db: Session, file_id: str, min_confidence: float | None = None
) -> list[Detection]:
    """
    Get all detections for a file.

    Args:
        file_id: File ID to get detections for
        min_confidence: Optional minimum confidence threshold (0.0-1.0)

    Returns empty list if no detections exist.
    """
    query = select(Detection).where(Detection.file_id == file_id)
    if min_confidence is not None:
        query = query.where(Detection.confidence >= min_confidence)
    query = query.order_by(Detection.confidence.desc())
    result = db.execute(query)
    return list(result.scalars().all())


def get_detections_by_job(db: Session, job_id: str) -> list[Detection]:
    """
    Get all detections created by a job.

    Returns empty list if no detections exist.
    Useful for stats and summaries.
    """
    query = select(Detection).where(Detection.job_id == job_id)
    result = db.execute(query)
    return list(result.scalars().all())


def create_detection(db: Session, detection: DetectionCreate) -> Detection:
    """
    Create a single detection.

    Crashes if database constraint violated (e.g., invalid file_id).
    This is intentional - we want to surface errors immediately.
    """
    db_detection = Detection(
        file_id=detection.file_id,
        job_id=detection.job_id,
        category=detection.category,
        confidence=detection.confidence,
        bbox_x=detection.bbox_x,
        bbox_y=detection.bbox_y,
        bbox_width=detection.bbox_width,
        bbox_height=detection.bbox_height,
        label=detection.label,
        label_confidence=detection.label_confidence,
        frame_number=detection.frame_number,
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection


def create_detections_bulk(
    db: Session, detections: list[DetectionCreate]
) -> list[Detection]:
    """
    Create multiple detections in a single transaction.

    More efficient than creating one at a time.
    Crashes if any detection violates database constraints.

    Args:
        detections: List of detection data to create

    Returns:
        List of created Detection objects
    """
    db_detections = [
        Detection(
            file_id=detection.file_id,
            job_id=detection.job_id,
            category=detection.category,
            confidence=detection.confidence,
            bbox_x=detection.bbox_x,
            bbox_y=detection.bbox_y,
            bbox_width=detection.bbox_width,
            bbox_height=detection.bbox_height,
            label=detection.label,
            label_confidence=detection.label_confidence,
            frame_number=detection.frame_number,
        )
        for detection in detections
    ]

    db.add_all(db_detections)
    db.commit()

    # Refresh all objects to get IDs and timestamps
    for detection in db_detections:
        db.refresh(detection)

    return db_detections


def get_detection_stats_by_job(db: Session, job_id: str) -> dict[str, int]:
    """
    Get detection statistics for a job.

    Returns counts by category.
    """
    detections = get_detections_by_job(db, job_id)

    stats: dict[str, int] = {
        "total": len(detections),
        "animal": 0,
        "person": 0,
        "vehicle": 0,
    }

    for detection in detections:
        category = detection.category.lower()
        if category in stats:
            stats[category] += 1

    return stats


def get_detection_stats_by_file(db: Session, file_id: str) -> dict[str, int]:
    """
    Get detection statistics for a file.

    Returns counts by category.
    """
    detections = get_detections_by_file(db, file_id)

    stats: dict[str, int] = {
        "total": len(detections),
        "animal": 0,
        "person": 0,
        "vehicle": 0,
    }

    for detection in detections:
        category = detection.category.lower()
        if category in stats:
            stats[category] += 1

    return stats


def create_human_detection(db: Session, data: DetectionCreateHuman) -> Detection:
    """
    Create a human-drawn detection.

    Sets job_id=None, classification_method="human", confidence=1.0.
    Resolves label_taxonomy_id + scientific_name so the new detection
    shares the same taxonomy row (and therefore the same display color)
    as other detections with the same label / builtin category.

    Created **verified**. A box someone drew by hand is the strongest
    signal there is, stronger than any model output, so it gets the same
    protection as a box they confirmed: postprocessing, rollup and
    smoothing all skip verified detections, and only verified rows keep
    their `original_label` out of the machine-final mirror at the end of
    `update_database_from_smoothed_results`. Left unverified it was the
    one human decision the pipeline was free to overwrite.
    """
    now = datetime.now(UTC)
    db_detection = Detection(
        file_id=data.file_id,
        job_id=None,
        category=data.category,
        confidence=1.0,
        bbox_x=data.bbox_x,
        bbox_y=data.bbox_y,
        bbox_width=data.bbox_width,
        bbox_height=data.bbox_height,
        label=data.label,
        label_confidence=1.0 if data.label else None,
        classification_method="human",
        frame_number=data.frame_number,
        verified=True,
        verified_at_utc=now,
    )
    db.add(db_detection)
    db.flush()  # populate id so _resolve_detection_taxonomy can find the project

    if data.label:
        db_detection.label_taxonomy_id = _resolve_detection_taxonomy(
            db, db_detection, data.label
        )
        from app.ml.taxonomic_rollup import resolve_label_names
        from app.models.label_taxonomy import LabelTaxonomy

        tax = (
            db.query(LabelTaxonomy).get(db_detection.label_taxonomy_id)
            if db_detection.label_taxonomy_id
            else None
        )
        (
            db_detection.common_name,
            db_detection.scientific_name,
        ) = resolve_label_names(data.label, tax, data.category)
    else:
        # Unclassified (MD-only) — share the builtin taxonomy row so
        # this detection gets the same color as the MD-produced ones.
        from app.ml.taxonomy_db import BUILTIN_MODEL_ID
        from app.models.label_taxonomy import LabelTaxonomy

        builtin = (
            db.query(LabelTaxonomy)
            .filter(
                LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID,
                LabelTaxonomy.name == data.category,
            )
            .first()
        )
        if builtin:
            db_detection.label_taxonomy_id = builtin.id
            db_detection.common_name = builtin.common_name
            db_detection.scientific_name = builtin.scientific_name

    db.commit()
    db.refresh(db_detection)
    return db_detection


def update_detection(db: Session, detection_id: str, update: DetectionUpdate) -> Detection | None:
    """
    Partial update of a detection.

    Sets classification_method to "human" when label is edited.
    """
    detection = get_detection(db, detection_id)
    if detection is None:
        return None

    if update.category is not None:
        detection.category = update.category
    if update.bbox_x is not None:
        detection.bbox_x = update.bbox_x
    if update.bbox_y is not None:
        detection.bbox_y = update.bbox_y
    if update.bbox_width is not None:
        detection.bbox_width = update.bbox_width
    if update.bbox_height is not None:
        detection.bbox_height = update.bbox_height
    if "label" in update.model_fields_set:
        detection.label = update.label
        detection.label_taxonomy_id = _resolve_detection_taxonomy(
            db, detection, update.label
        )
        detection.classification_method = "human"
        # Resolve both names from the taxonomy row (single source of truth).
        if update.label:
            from app.ml.taxonomic_rollup import resolve_label_names
            from app.models.label_taxonomy import LabelTaxonomy

            tax = (
                db.query(LabelTaxonomy).get(detection.label_taxonomy_id)
                if detection.label_taxonomy_id
                else None
            )
            (
                detection.common_name,
                detection.scientific_name,
            ) = resolve_label_names(update.label, tax, detection.category)
        else:
            detection.scientific_name = None
            detection.common_name = None
        # A human-assigned label has no model softmax score, so stamp 1.0
        # (matches bulk relabel) rather than leaving the replaced label's
        # stale score. Cleared label -> no confidence. An explicit
        # label_confidence in the payload still overrides below.
        detection.label_confidence = 1.0 if update.label else None
    # When category changes to a builtin (person/vehicle/animal) without a
    # label, resolve taxonomy from the category so scientific_name and the FK
    # are set correctly.
    if update.category and not detection.label:
        from app.ml.taxonomy_db import BUILTIN_MODEL_ID
        from app.models.label_taxonomy import LabelTaxonomy

        builtin = (
            db.query(LabelTaxonomy)
            .filter(
                LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID,
                LabelTaxonomy.name == update.category,
            )
            .first()
        )
        if builtin:
            detection.label_taxonomy_id = builtin.id
            detection.common_name = builtin.common_name
            detection.scientific_name = builtin.scientific_name
    if update.label_confidence is not None:
        detection.label_confidence = update.label_confidence

    db.commit()
    db.refresh(detection)
    return detection


def delete_detection(db: Session, detection_id: str) -> bool:
    """
    Delete a detection.

    Returns True if deleted, False if detection doesn't exist.
    """
    db_detection = get_detection(db, detection_id)
    if db_detection is None:
        return False

    db.delete(db_detection)
    db.commit()
    return True


def delete_detections_by_file(db: Session, file_id: str) -> int:
    """
    Delete all detections for a file.

    Returns count of detections deleted.
    Useful for re-running detection on a file.
    """
    detections = get_detections_by_file(db, file_id)
    count = len(detections)

    for detection in detections:
        db.delete(detection)

    db.commit()
    return count


def _get_project_id_for_detection(db: Session, detection: Detection) -> str | None:
    """Resolve project_id from Detection → File → Deployment."""
    row = (
        db.query(Deployment.project_id)
        .join(File, File.deployment_id == Deployment.id)
        .filter(File.id == detection.file_id)
        .first()
    )
    return row[0] if row else None


def _resolve_detection_taxonomy(
    db: Session, detection: Detection, label_name: str | None
) -> str | None:
    """Look up or auto-create the label_taxonomy_id for a relabeled detection.

    If no existing taxonomy entry matches, creates a custom entry with
    level='unknown' so every label has a corresponding taxonomy row.
    """
    if not label_name:
        return None
    project_id = _get_project_id_for_detection(db, detection)
    if not project_id:
        return None

    from app.ml.taxonomy_db import resolve_taxonomy_id

    taxonomy_id = resolve_taxonomy_id(label_name, project_id, db)
    if taxonomy_id:
        return taxonomy_id

    # Auto-create a custom taxonomy entry for this label
    from app.ml.taxonomic_rollup import format_common_name
    from app.models.label_taxonomy import LabelTaxonomy

    new_entry = LabelTaxonomy(
        is_custom=True,
        project_id=project_id,
        level="unknown",
        name=label_name,
        classification_model_id="",
        common_name=format_common_name(label_name) if label_name else label_name,
        scientific_name=label_name[0].upper() + label_name[1:]
        if label_name
        else label_name,
    )
    db.add(new_entry)
    db.flush()
    return new_entry.id
