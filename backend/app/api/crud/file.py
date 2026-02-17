"""
CRUD operations for files.
"""

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.api.schemas.file import FileUpdate
from app.models import Deployment, Detection, File


def get_files(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    observation_type: str | None = None,
) -> list[File]:
    """
    Get all files with pagination.

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Number of records to return
        observation_type: Optional filter by observation type

    Returns:
        List of files
    """
    query = db.query(File)
    if observation_type:
        query = query.filter(File.observation_type == observation_type)
    return query.order_by(File.timestamp.desc()).offset(skip).limit(limit).all()


def get_files_by_deployment(
    db: Session,
    deployment_id: str,
    skip: int = 0,
    limit: int = 100,
    observation_type: str | None = None,
) -> list[File]:
    """
    Get files by deployment ID.

    Args:
        db: Database session
        deployment_id: Deployment ID
        skip: Number of records to skip
        limit: Number of records to return
        observation_type: Optional filter by observation type

    Returns:
        List of files
    """
    query = db.query(File).filter(File.deployment_id == deployment_id)
    if observation_type:
        query = query.filter(File.observation_type == observation_type)
    return query.order_by(File.timestamp.desc()).offset(skip).limit(limit).all()


def get_files_by_project(
    db: Session,
    project_id: str,
    skip: int = 0,
    limit: int = 100,
    observation_type: str | None = None,
) -> list[File]:
    """
    Get files by project ID.

    Args:
        db: Database session
        project_id: Project ID
        skip: Number of records to skip
        limit: Number of records to return
        observation_type: Optional filter by observation type

    Returns:
        List of files
    """
    query = (
        db.query(File)
        .join(Deployment)
        .join(Deployment.site)
        .filter(Deployment.site.has(project_id=project_id))
    )
    if observation_type:
        query = query.filter(File.observation_type == observation_type)
    return query.order_by(File.timestamp.desc()).offset(skip).limit(limit).all()


def get_file_with_detections(db: Session, file_id: str) -> File | None:
    """
    Get file by ID with detections loaded.

    Args:
        db: Database session
        file_id: File ID

    Returns:
        File with detections or None if not found
    """
    return (
        db.query(File)
        .options(joinedload(File.detections))
        .filter(File.id == file_id)
        .first()
    )


def update_file(db: Session, file_id: str, update: FileUpdate) -> File | None:
    """
    Update a file's verification status and/or notes.

    Sets verified_at to current time when verified changes to True,
    clears it when verified changes to False.
    """
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return None

    if update.verified is not None:
        if update.verified and not file.verified:
            file.verified = True
            file.verified_at = datetime.utcnow()
        elif not update.verified and file.verified:
            file.verified = False
            file.verified_at = None

    if update.notes is not None:
        file.notes = update.notes

    if update.favorited is not None:
        file.favorited = update.favorited

    db.commit()
    db.refresh(file)
    return file


def get_observation_type_stats(
    db: Session, project_id: str
) -> dict[str, int]:
    """
    Get observation type counts for a project.

    Args:
        db: Database session
        project_id: Project ID

    Returns:
        Dict mapping observation_type -> count
    """
    rows = (
        db.query(File.observation_type, func.count(File.id))
        .join(Deployment)
        .join(Deployment.site)
        .filter(Deployment.site.has(project_id=project_id))
        .group_by(File.observation_type)
        .all()
    )
    return {obs_type: count for obs_type, count in rows}


def recalculate_observation_type(db: Session, file_id: str) -> None:
    """
    Re-derive observation_type from current detections.

    Priority: animal > human > vehicle > blank.
    Called after detection create/update/delete.
    """
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        return

    detections = (
        db.query(Detection)
        .filter(Detection.file_id == file_id)
        .all()
    )

    if not detections:
        file.observation_type = "blank"
    else:
        # Map detection categories to observation types
        category_map = {"animal": "animal", "person": "human", "vehicle": "vehicle"}
        priority = {"animal": 4, "human": 3, "vehicle": 2}

        best_type = "blank"
        best_priority = 0
        for d in detections:
            obs = category_map.get(d.category, "unknown")
            p = priority.get(obs, 0)
            if p > best_priority:
                best_priority = p
                best_type = obs

        file.observation_type = best_type

    db.commit()
