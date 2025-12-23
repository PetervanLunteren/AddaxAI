"""
CRUD operations for DeploymentQueue model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.deployment_queue import DeploymentQueueCreate
from app.models import DeploymentQueue


def get_queue_entries(db: Session, project_id: str, status: str | None = None) -> list[DeploymentQueue]:
    """
    Get all queue entries for a project.

    Optionally filter by status.
    Returns empty list if no entries exist.
    """
    query = select(DeploymentQueue).where(DeploymentQueue.project_id == project_id)

    if status:
        query = query.where(DeploymentQueue.status == status)

    query = query.order_by(DeploymentQueue.created_at.asc())
    result = db.execute(query)
    return list(result.scalars().all())


def get_queue_entry(db: Session, entry_id: str) -> DeploymentQueue | None:
    """
    Get queue entry by ID.

    Returns None if entry doesn't exist.
    """
    result = db.execute(select(DeploymentQueue).where(DeploymentQueue.id == entry_id))
    return result.scalar_one_or_none()


def create_queue_entry(db: Session, entry: DeploymentQueueCreate) -> DeploymentQueue:
    """
    Create a new queue entry.

    Crashes if database constraint violated.
    This is intentional - we want to surface errors immediately.
    """
    db_entry = DeploymentQueue(
        project_id=entry.project_id,
        folder_path=entry.folder_path,
        site_id=entry.site_id,
        detection_model_id=entry.detection_model_id,
        classification_model_id=entry.classification_model_id,
        species_list=entry.species_list,
        status="pending",
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry


def update_queue_status(
    db: Session,
    entry_id: str,
    status: str,
    error: str | None = None,
    deployment_id: str | None = None
) -> DeploymentQueue | None:
    """
    Update queue entry status.

    Returns None if entry doesn't exist.
    """
    db_entry = get_queue_entry(db, entry_id)
    if db_entry is None:
        return None

    db_entry.status = status

    if error:
        db_entry.error = error

    if deployment_id:
        db_entry.deployment_id = deployment_id

    if status in ["completed", "failed"]:
        db_entry.processed_at = datetime.utcnow()

    db.commit()
    db.refresh(db_entry)
    return db_entry


def delete_queue_entry(db: Session, entry_id: str) -> bool:
    """
    Delete a queue entry.

    Returns True if deleted, False if entry doesn't exist.
    """
    db_entry = get_queue_entry(db, entry_id)
    if db_entry is None:
        return False

    db.delete(db_entry)
    db.commit()
    return True
