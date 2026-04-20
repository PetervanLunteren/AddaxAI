"""
CRUD operations for Project model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.crud.statistics import get_trap_nights
from app.api.schemas.project import ProjectCreate, ProjectUpdate
from app.models import Deployment, Event, File, Project, Site
from app.models.event_observation import EventObservation


def get_projects(db: Session) -> list[Project]:
    """
    Get all projects.

    Returns empty list if no projects exist.
    """
    result = db.execute(select(Project).order_by(Project.created_at_utc.desc()))
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
        embedding_model_id=project.embedding_model_id,
        excluded_classes=project.excluded_classes if project.excluded_classes else [],
        shortcut_labels=project.shortcut_labels if project.shortcut_labels else {},
        country_code=project.country_code,
        state_code=project.state_code,
        timezone=project.timezone,
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
    Get statistics for a single project.

    Returns dict with counts, or None if project doesn't exist.
    """
    db_project = get_project(db, project_id)
    if db_project is None:
        return None

    # Count sites
    site_count = (
        db.scalar(
            select(func.count(Site.id)).where(Site.project_id == project_id)
        )
        or 0
    )

    # Count deployments
    deployment_count = (
        db.scalar(
            select(func.count(Deployment.id)).where(
                Deployment.project_id == project_id
            )
        )
        or 0
    )

    # Count files
    file_count = (
        db.scalar(
            select(func.count(File.id))
            .join(Deployment, File.deployment_id == Deployment.id)
            .where(Deployment.project_id == project_id)
        )
        or 0
    )

    # Count observations (sum of MaxN from event_observations)
    observation_count = (
        db.scalar(
            select(func.coalesce(func.sum(EventObservation.max_n), 0))
            .join(Event, Event.id == EventObservation.event_id)
            .join(Deployment, Event.deployment_id == Deployment.id)
            .where(Deployment.project_id == project_id)
        )
        or 0
    )

    trap_nights = get_trap_nights(db, project_id)

    return {
        "site_count": site_count,
        "deployment_count": deployment_count,
        "file_count": file_count,
        "observation_count": observation_count,
        "trap_nights": trap_nights,
    }


def get_all_projects_stats(db: Session) -> dict[str, dict[str, int]]:
    """
    Get statistics for all projects in bulk.

    Returns a dict keyed by project_id, each containing counts for
    sites, deployments, files, observations, and trap nights.
    Deployments with no site (site_id=None) still belong to their
    project and are counted.
    """
    site_rows = db.execute(
        select(Site.project_id, func.count(Site.id)).group_by(Site.project_id)
    ).all()
    dep_rows = db.execute(
        select(Deployment.project_id, func.count(Deployment.id)).group_by(
            Deployment.project_id
        )
    ).all()
    file_rows = db.execute(
        select(Deployment.project_id, func.count(File.id))
        .join(File, File.deployment_id == Deployment.id)
        .group_by(Deployment.project_id)
    ).all()
    obs_rows = db.execute(
        select(
            Deployment.project_id,
            func.coalesce(func.sum(EventObservation.max_n), 0),
        )
        .join(Event, Event.deployment_id == Deployment.id)
        .join(EventObservation, EventObservation.event_id == Event.id)
        .group_by(Deployment.project_id)
    ).all()

    stats: dict[str, dict[str, int]] = {}

    def _bucket(project_id: str) -> dict[str, int]:
        return stats.setdefault(
            project_id,
            {
                "site_count": 0,
                "deployment_count": 0,
                "file_count": 0,
                "observation_count": 0,
                "trap_nights": 0,
            },
        )

    for project_id, count in site_rows:
        _bucket(project_id)["site_count"] = int(count)
    for project_id, count in dep_rows:
        _bucket(project_id)["deployment_count"] = int(count)
    for project_id, count in file_rows:
        _bucket(project_id)["file_count"] = int(count)
    for project_id, count in obs_rows:
        _bucket(project_id)["observation_count"] = int(count)

    for project_id in list(stats.keys()):
        stats[project_id]["trap_nights"] = get_trap_nights(db, project_id)

    return stats
