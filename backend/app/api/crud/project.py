"""
CRUD operations for Project model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from app.api.crud.statistics import get_trap_nights
from app.api.schemas.project import ProjectCreate, ProjectUpdate
from app.models import Deployment, Detection, File, Project, Site


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
        embedding_model_id=project.embedding_model_id,
        excluded_classes=project.excluded_classes if project.excluded_classes else [],
        shortcut_labels=project.shortcut_labels if project.shortcut_labels else {},
        country_code=project.country_code,
        state_code=project.state_code,
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

    # Count detections (respect project threshold; verified always included)
    threshold = db_project.detection_threshold or 0.0
    detection_count = (
        db.scalar(
            select(func.count(Detection.id))
            .join(File)
            .join(Deployment)
            .join(Site)
            .where(Site.project_id == project_id)
            .where(
                or_(
                    Detection.confidence >= threshold,
                    Detection.verified == True,  # noqa: E712
                )
            )
        )
        or 0
    )

    trap_nights = get_trap_nights(db, project_id)

    return {
        "site_count": site_count,
        "deployment_count": deployment_count,
        "file_count": file_count,
        "detection_count": detection_count,
        "trap_nights": trap_nights,
    }


def get_all_projects_stats(db: Session) -> dict[str, dict[str, int]]:
    """
    Get statistics for all projects in bulk.

    Returns a dict keyed by project_id, each containing counts for
    sites, deployments, files, detections, and trap nights.
    """
    # Single query for site, deployment, file, and detection counts per project.
    # Detection count respects each project's threshold; verified always included.
    meets_threshold = case(
        (
            or_(
                Detection.confidence >= func.coalesce(
                    Project.detection_threshold, 0.0
                ),
                Detection.verified == True,  # noqa: E712
            ),
            1,
        ),
        else_=0,
    )
    rows = db.execute(
        select(
            Site.project_id,
            func.count(func.distinct(Site.id)).label("site_count"),
            func.count(func.distinct(Deployment.id)).label("deployment_count"),
            func.count(func.distinct(File.id)).label("file_count"),
            func.sum(meets_threshold).label("detection_count"),
        )
        .select_from(Site)
        .join(Project, Project.id == Site.project_id)
        .outerjoin(Deployment, Deployment.site_id == Site.id)
        .outerjoin(File, File.deployment_id == Deployment.id)
        .outerjoin(Detection, Detection.file_id == File.id)
        .group_by(Site.project_id)
    ).all()

    stats: dict[str, dict[str, int]] = {}
    project_ids_with_sites: list[str] = []

    for row in rows:
        project_id = row.project_id
        project_ids_with_sites.append(project_id)
        stats[project_id] = {
            "site_count": row.site_count,
            "deployment_count": row.deployment_count,
            "file_count": row.file_count,
            "detection_count": int(row.detection_count or 0),
            "trap_nights": 0,
        }

    # Compute trap nights per project (reuses existing calculation)
    for project_id in project_ids_with_sites:
        stats[project_id]["trap_nights"] = get_trap_nights(db, project_id)

    return stats
