"""
CRUD operations for Project model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas.project import ProjectCreate, ProjectUpdate
from app.models import Deployment, File, Project, Site


def get_projects(db: Session) -> list[Project]:
    """
    Get all projects.

    Returns empty list if no projects exist.
    """
    result = db.execute(select(Project).order_by(Project.created_at.desc()))
    return list(result.scalars().all())


def get_project(db: Session, project_id: str) -> Project | None:
    """
    Get project by ID.

    Returns None if project doesn't exist.
    """
    result = db.execute(select(Project).where(Project.id == project_id))
    return result.scalar_one_or_none()


def create_project(db: Session, project: ProjectCreate) -> Project:
    """
    Create a new project.

    Crashes if database constraint violated (e.g., duplicate name).
    This is intentional - we want to surface errors immediately.
    """
    db_project = Project(
        name=project.name,
        description=project.description,
        detection_model_id=project.detection_model_id,
        classification_model_id=project.classification_model_id,
        excluded_classes=project.excluded_classes if project.excluded_classes else [],
        detection_threshold=project.detection_threshold,
        event_smoothing=project.event_smoothing,
        taxonomic_rollup=project.taxonomic_rollup,
        taxonomic_rollup_threshold=project.taxonomic_rollup_threshold,
        independence_interval=project.independence_interval,
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


def update_project(db: Session, project_id: str, project: ProjectUpdate) -> Project | None:
    """
    Update an existing project.

    Returns None if project doesn't exist.
    Only updates fields that are provided (not None).
    Crashes if database constraint violated.
    """
    db_project = get_project(db, project_id)
    if db_project is None:
        return None

    # Only update provided fields
    update_data = project.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_project, field, value)

    db.commit()
    db.refresh(db_project)
    return db_project


def delete_project(db: Session, project_id: str) -> bool:
    """
    Delete a project.

    Returns True if deleted, False if project doesn't exist.
    Cascades to all related sites, deployments, files, etc.
    """
    db_project = get_project(db, project_id)
    if db_project is None:
        return False

    db.delete(db_project)
    db.commit()
    return True


def get_project_stats(db: Session, project_id: str) -> dict[str, int] | None:
    """
    Get statistics for a project.

    Returns dict with counts, or None if project doesn't exist.
    """
    db_project = get_project(db, project_id)
    if db_project is None:
        return None

    # Count sites
    site_count = db.scalar(select(func.count(Site.id)).where(Site.project_id == project_id)) or 0

    # Count deployments
    deployment_count = (
        db.scalar(
            select(func.count(Deployment.id))
            .join(Site)
            .where(Site.project_id == project_id)
        )
        or 0
    )

    # Count files
    file_count = (
        db.scalar(
            select(func.count(File.id))
            .join(Deployment)
            .join(Site)
            .where(Site.project_id == project_id)
        )
        or 0
    )

    # TODO: Count detections (model not implemented yet)
    detection_count = 0

    return {
        "site_count": site_count,
        "deployment_count": deployment_count,
        "file_count": file_count,
        "detection_count": detection_count,
    }
