"""
MaxN calculation and storage for event observations.

MaxN is the maximum number of individuals of a species visible in any
single image within an event. Calculated per-species, stored in the
event_observations table.
"""

import uuid

from sqlalchemy import delete, func, or_
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Deployment, Detection, Event, File, Project
from app.models.event import event_files
from app.models.event_observation import EventObservation

logger = get_logger(__name__)


def _threshold_clause(threshold: float):
    """Detection threshold filter: confidence >= threshold OR verified."""
    return or_(
        Detection.confidence >= threshold,
        Detection.verified == True,  # noqa: E712
    )


def calculate_max_n_for_event(
    db: Session,
    event_id: str,
    detection_threshold: float,
) -> list[EventObservation]:
    """
    Calculate and store MaxN per label for a single event.

    Algorithm:
    1. Get all detections in this event's files that pass the threshold
    2. Group by (file_id, effective_label) to count detections per file,
       summing detection confidence per group
    3. For each label, find the file with the maximum count (= MaxN).
       Ties are broken by the highest summed detection confidence, so the
       chosen frame is the one where the model is most certain about the
       animals it sees.
    4. Replace existing EventObservation rows with new ones

    Returns the created EventObservation rows.
    """
    # Group by label_taxonomy_id (authoritative), falling back to
    # COALESCE(label, category) for display string.
    effective_label = func.coalesce(Detection.label, Detection.category)

    # Count detections per file per taxonomy_id, plus summed confidence
    # for tie-breaking when multiple files share the same MaxN count.
    counts = (
        db.query(
            Detection.file_id,
            Detection.label_taxonomy_id,
            effective_label.label("eff_label"),
            Detection.category,
            func.count(Detection.id).label("det_count"),
            func.sum(Detection.confidence).label("conf_sum"),
        )
        .join(File, File.id == Detection.file_id)
        .join(event_files, event_files.c.file_id == File.id)
        .filter(event_files.c.event_id == event_id)
        .filter(_threshold_clause(detection_threshold))
        .group_by(
            Detection.file_id,
            Detection.label_taxonomy_id,
            effective_label,
            Detection.category,
        )
        .all()
    )

    if not counts:
        # No detections passing threshold: clear any existing observations
        db.execute(
            delete(EventObservation).where(
                EventObservation.event_id == event_id
            )
        )
        return []

    # Find MaxN per taxonomy_id (or label string as fallback key).
    # Score is (det_count, conf_sum) so that ties on count are broken by
    # the file with the highest summed detection confidence.
    max_n_per_key: dict[str, dict] = {}
    for file_id, taxonomy_id, label, category, det_count, conf_sum in counts:
        key = taxonomy_id or label
        new_score = (det_count, conf_sum)
        existing = max_n_per_key.get(key)
        if existing is None or new_score > (existing["count"], existing["conf_sum"]):
            max_n_per_key[key] = {
                "count": det_count,
                "conf_sum": conf_sum,
                "file_id": file_id,
                "category": category,
                "label": label,
                "taxonomy_id": taxonomy_id,
            }

    # Replace existing observations for this event
    db.execute(
        delete(EventObservation).where(
            EventObservation.event_id == event_id
        )
    )

    observations = []
    for data in max_n_per_key.values():
        obs = EventObservation(
            id=str(uuid.uuid4()),
            event_id=event_id,
            label=data["label"],
            label_taxonomy_id=data["taxonomy_id"],
            category=data["category"],
            max_n=data["count"],
            max_n_file_id=data["file_id"],
        )
        db.add(obs)
        observations.append(obs)

    return observations


def recalculate_max_n_for_project(
    db: Session,
    project_id: str,
) -> int:
    """
    Recalculate MaxN for all events in a project.

    Returns the total number of EventObservation rows created.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    threshold = project.detection_threshold

    # Get all event IDs for this project
    event_ids = (
        db.query(Event.id)
        .join(Deployment, Deployment.id == Event.deployment_id)
        .filter(Deployment.project_id == project_id)
        .all()
    )

    total_observations = 0
    for (event_id,) in event_ids:
        obs = calculate_max_n_for_event(db, event_id, threshold)
        total_observations += len(obs)

    logger.info(
        f"Recalculated MaxN for project {project_id}: "
        f"{len(event_ids)} events, {total_observations} observations"
    )
    return total_observations


def recalculate_max_n_for_events(
    db: Session,
    event_ids: list[str],
    detection_threshold: float,
) -> None:
    """Recalculate MaxN for specific events (after verify/relabel)."""
    for event_id in event_ids:
        calculate_max_n_for_event(db, event_id, detection_threshold)


def get_event_ids_for_detections(
    db: Session,
    detection_ids: list[str],
) -> list[str]:
    """Get event IDs that contain files with the given detections."""
    rows = (
        db.query(event_files.c.event_id)
        .join(File, File.id == event_files.c.file_id)
        .join(Detection, Detection.file_id == File.id)
        .filter(Detection.id.in_(detection_ids))
        .distinct()
        .all()
    )
    return [row[0] for row in rows]


def get_thumbnail_file_id(db: Session, event_id: str) -> str | None:
    """Get the max_n_file_id of the dominant species (highest max_n)."""
    obs = (
        db.query(EventObservation.max_n_file_id)
        .filter(EventObservation.event_id == event_id)
        .order_by(EventObservation.max_n.desc())
        .first()
    )
    return obs[0] if obs else None


def get_max_n_frames(db: Session, event_id: str) -> list[dict]:
    """Get all MaxN frames for an event, ordered by max_n descending."""
    rows = (
        db.query(
            EventObservation.max_n_file_id,
            EventObservation.label,
            EventObservation.max_n,
            EventObservation.label_taxonomy_id,
        )
        .filter(EventObservation.event_id == event_id)
        .filter(EventObservation.max_n_file_id.isnot(None))
        .order_by(EventObservation.max_n.desc())
        .all()
    )
    return [
        {
            "file_id": row[0],
            "label": row[1],
            "max_n": row[2],
            "label_taxonomy_id": row[3],
        }
        for row in rows
    ]


def get_project_threshold_for_detections(
    db: Session,
    detection_ids: list[str],
) -> float:
    """Get the project detection_threshold for the given detections."""
    row = (
        db.query(Project.detection_threshold)
        .join(Deployment, Deployment.project_id == Project.id)
        .join(File, File.deployment_id == Deployment.id)
        .join(Detection, Detection.file_id == File.id)
        .filter(Detection.id.in_(detection_ids))
        .first()
    )
    return row[0] if row else 0.0
