"""
CRUD operations for Deployment model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas.deployment import DeploymentCreate, DeploymentUpdate
from app.models import Deployment, Event, File


def get_deployments(db: Session, site_id: str | None = None) -> list[Deployment]:
    """
    Get all deployments, optionally filtered by site_id.

    Returns empty list if no deployments exist.
    """
    query = select(Deployment).order_by(Deployment.created_at.desc())
    if site_id:
        query = query.where(Deployment.site_id == site_id)
    result = db.execute(query)
    return list(result.scalars().all())


def get_deployment(db: Session, deployment_id: str) -> Deployment | None:
    """
    Get deployment by ID.

    Returns None if deployment doesn't exist.
    """
    result = db.execute(select(Deployment).where(Deployment.id == deployment_id))
    return result.scalar_one_or_none()


def create_deployment(db: Session, deployment: DeploymentCreate) -> Deployment:
    """
    Create a new deployment.

    Crashes if database constraint violated (e.g., invalid site_id).
    This is intentional - we want to surface errors immediately.
    """
    db_deployment = Deployment(
        site_id=deployment.site_id,
        folder_path=deployment.folder_path,
        start_date=deployment.start_date,
        end_date=deployment.end_date,
        camera_model=deployment.camera_model,
        camera_serial=deployment.camera_serial,
        notes=deployment.notes,
    )
    db.add(db_deployment)
    db.commit()
    db.refresh(db_deployment)
    return db_deployment


def update_deployment(
    db: Session, deployment_id: str, deployment: DeploymentUpdate
) -> Deployment | None:
    """
    Update an existing deployment.

    Returns None if deployment doesn't exist.
    Only updates fields that are provided (not None).
    Crashes if database constraint violated.

    When folder_path is updated (re-linking), also updates last_validated_at.
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return None

    # Only update provided fields
    update_data = deployment.model_dump(exclude_unset=True)

    # If folder_path is being updated, update validation timestamp
    if "folder_path" in update_data and update_data["folder_path"] is not None:
        db_deployment.folder_status = "valid"
        db_deployment.last_validated_at = datetime.utcnow()

    for field, value in update_data.items():
        setattr(db_deployment, field, value)

    db.commit()
    db.refresh(db_deployment)
    return db_deployment


def delete_deployment(db: Session, deployment_id: str) -> bool:
    """
    Delete a deployment.

    Returns True if deleted, False if deployment doesn't exist.
    Cascades to all related files and events.
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return False

    db.delete(db_deployment)
    db.commit()
    return True


def get_deployment_stats(db: Session, deployment_id: str) -> dict[str, int] | None:
    """
    Get statistics for a deployment.

    Returns dict with counts, or None if deployment doesn't exist.
    """
    db_deployment = get_deployment(db, deployment_id)
    if db_deployment is None:
        return None

    # Count files
    file_count = (
        db.scalar(
            select(func.count(File.id)).where(File.deployment_id == deployment_id)
        )
        or 0
    )

    # Count events
    event_count = (
        db.scalar(
            select(func.count(Event.id)).where(Event.deployment_id == deployment_id)
        )
        or 0
    )

    # TODO: Count detections (model not fully implemented yet)
    detection_count = 0

    return {
        "file_count": file_count,
        "event_count": event_count,
        "detection_count": detection_count,
    }
