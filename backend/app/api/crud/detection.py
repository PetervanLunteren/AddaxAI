"""
CRUD operations for Detection model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas.detection import DetectionCreate
from app.models import Detection


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
        species=detection.species,
        species_confidence=detection.species_confidence,
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
            species=detection.species,
            species_confidence=detection.species_confidence,
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
