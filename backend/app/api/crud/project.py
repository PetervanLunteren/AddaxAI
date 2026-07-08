"""
CRUD operations for Project model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.crud.statistics import get_trap_nights
from app.api.schemas.project import (
    ProjectCreate,
    ProjectDuplicate,
    ProjectMode,
    ProjectUpdate,
)
from app.models import Deployment, Event, File, Project, Site
from app.models.deployment_queue import DeploymentQueue
from app.models.event_observation import EventObservation

# Processing settings carried over wholesale when copy_settings is on. The
# user-chosen fields (name, description, classification model, label selection)
# are set from the request instead. Omitting these when copy_settings is off
# lets the model's column defaults apply, matching a fresh project.
_DUPLICATE_SETTINGS_COLUMNS = (
    "detection_model_id",
    "embedding_model_id",
    "timezone",
    "shortcut_labels",
    "video_fps",
    "detection_threshold",
    "event_smoothing",
    "smoothing_strength",
    "taxonomic_rollup",
    "independence_interval",
    "min_cluster_size",
    "min_samples",
    "detection_batch_size",
    "classification_batch_size",
    "embedding_batch_size",
)


def get_projects(
    db: Session, mode: ProjectMode | None = None
) -> list[Project]:
    """
    Get projects, optionally filtered by mode.

    When `mode` is None, returns every project regardless of workflow
    mode. Callers that render the user-facing Research projects list
    pass `mode='research'` so folder runs stay hidden. Returns empty
    list if no projects match.
    """
    stmt = select(Project).order_by(Project.created_at_utc.desc())
    if mode is not None:
        stmt = stmt.where(Project.mode == mode)
    result = db.execute(stmt)
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
        independence_interval=project.independence_interval,
        mode=project.mode,
        folder_run_state=project.folder_run_state,
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


def duplicate_project(
    db: Session, source_id: str, params: ProjectDuplicate
) -> Project | None:
    """Create a new project from an existing one's structure.

    Always copies the user-chosen fields (name, description, classification
    model, label selection). Copies processing settings when copy_settings,
    sites when copy_sites, and re-queues the source's deployments for
    reprocessing when copy_deployments. Analyzed results are never copied across
    projects (only the folders are queued). Returns None if the source is
    missing. Raises on a duplicate name (unique constraint), like create.
    """
    source = get_project(db, source_id)
    if source is None:
        return None

    new_project = Project(
        name=params.name,
        description=params.description,
        classification_model_id=params.classification_model_id,
        excluded_classes=list(params.excluded_classes or []),
        country_code=params.country_code,
        state_code=params.state_code,
        mode=source.mode,
    )
    if params.copy_settings:
        for col in _DUPLICATE_SETTINGS_COLUMNS:
            setattr(new_project, col, getattr(source, col))
    db.add(new_project)
    db.flush()  # assign new_project.id for the FK references below

    # Sites and deployments are independent. When deployments are copied
    # without sites the site_id_map stays empty, so the re-queued folders come
    # in without a site assignment (which the user can set later).
    site_id_map: dict[str, str] = {}
    if params.copy_sites:
        for site in source.sites:
            new_site = Site(
                project_id=new_project.id,
                name=site.name,
                latitude=site.latitude,
                longitude=site.longitude,
                elevation_m=site.elevation_m,
                habitat_type=site.habitat_type,
                notes=site.notes,
                tags=dict(site.tags or {}),
            )
            db.add(new_site)
            db.flush()
            site_id_map[site.id] = new_site.id

    if params.copy_deployments:
        for dep in source.deployments:
            if not dep.folder_path:
                continue
            db.add(
                DeploymentQueue(
                    project_id=new_project.id,
                    folder_path=dep.folder_path,
                    site_id=(
                        site_id_map.get(dep.site_id) if dep.site_id else None
                    ),
                    notes=dep.notes,
                    tags=dict(dep.tags or {}),
                    datetime_offset_seconds=dep.datetime_offset_seconds,
                    status="pending",
                )
            )

    db.commit()
    db.refresh(new_project)
    return new_project


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


def delete_folder_run(db: Session, project_id: str) -> bool:
    """
    Delete a folder-run project and its on-disk artifacts.

    Cascade-deletes the Project + its DeploymentQueue + Deployment(s) +
    Files + Detections + Events + EventObservations via SQLAlchemy
    relationships, then removes the per-project ``.addaxai/projects/
    <project_id>/`` cache folder from every source folder the project
    pointed at.

    Use this rather than ``delete_project`` for folder-run cleanup
    because the artifact cleanup is required to free the cache (PIL
    best frames, intermediate JSONs) the analysis worker dropped on
    disk during the previous run.

    Returns True on success, False when the project doesn't exist.
    Jobs are intentionally left alone — they have no FK to project
    and serve as a history record.
    """
    from app.api.crud.deployment import _delete_deployment_artifacts

    db_project = get_project(db, project_id)
    if db_project is None:
        return False

    # Capture every folder_path that might host a ``.addaxai`` cache
    # before the cascade fires. Deployments hold the path when analysis
    # has run; the queue entry holds it when the user picked the folder
    # but never ran analysis. Either way we'd rather try and find
    # nothing than miss a stale cache.
    folder_paths: set[str] = set()
    folder_paths.update(
        d.folder_path for d in db_project.deployments if d.folder_path
    )
    folder_paths.update(
        q.folder_path for q in db_project.deployment_queue if q.folder_path
    )

    db.delete(db_project)
    db.commit()

    for folder_path in folder_paths:
        _delete_deployment_artifacts(folder_path, project_id)
    return True


def reset_folder_run_data(db: Session, project_id: str) -> bool:
    """
    Reset a folder-run for re-analysis.

    Wipes the analysis output (deployments, files, detections, events,
    embeddings) and the on-disk ``.addaxai/projects/<project_id>/``
    cache, but keeps the project row and the queue entry so the run id
    survives across the re-run. The queue entry is moved back to
    ``status='pending'`` with cleared error / warning / processed-at /
    deployment-id fields so the existing ``POST /api/deployment-queue/
    process`` flow picks it up as if it had never run.

    Returns True on success, False when the project doesn't exist.

    Verified detections are destroyed by this operation. The caller
    must surface a destructive confirm dialog before invoking it.
    """
    from app.api.crud.deployment import _delete_deployment_artifacts

    db_project = get_project(db, project_id)
    if db_project is None:
        return False

    folder_paths: set[str] = set()
    folder_paths.update(
        d.folder_path for d in db_project.deployments if d.folder_path
    )
    folder_paths.update(
        q.folder_path for q in db_project.deployment_queue if q.folder_path
    )

    for deployment in list(db_project.deployments):
        db.delete(deployment)

    for queue_entry in db_project.deployment_queue:
        queue_entry.status = "pending"
        queue_entry.error = None
        queue_entry.warnings = None
        queue_entry.processed_at_utc = None
        queue_entry.deployment_id = None

    db_project.updated_at_utc = datetime.now(UTC)
    db.commit()

    for folder_path in folder_paths:
        _delete_deployment_artifacts(folder_path, project_id)
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


def get_all_projects_stats(
    db: Session, mode: ProjectMode | None = None
) -> dict[str, dict[str, int]]:
    """
    Get statistics for all projects in bulk.

    Returns a dict keyed by project_id, each containing counts for
    sites, deployments, files, observations, and trap nights.
    Deployments with no site (site_id=None) still belong to their
    project and are counted.

    When `mode` is set, only projects with that workflow mode are
    included. Folder runs typically have one auto-named deployment and
    no sites, so leaving them in the totals would skew the Research
    projects list display.
    """
    project_filter: list = []
    if mode is not None:
        project_filter.append(Project.mode == mode)

    site_rows = db.execute(
        select(Site.project_id, func.count(Site.id))
        .join(Project, Project.id == Site.project_id)
        .where(*project_filter)
        .group_by(Site.project_id)
    ).all()
    dep_rows = db.execute(
        select(Deployment.project_id, func.count(Deployment.id))
        .join(Project, Project.id == Deployment.project_id)
        .where(*project_filter)
        .group_by(Deployment.project_id)
    ).all()
    file_rows = db.execute(
        select(Deployment.project_id, func.count(File.id))
        .join(File, File.deployment_id == Deployment.id)
        .join(Project, Project.id == Deployment.project_id)
        .where(*project_filter)
        .group_by(Deployment.project_id)
    ).all()
    obs_rows = db.execute(
        select(
            Deployment.project_id,
            func.coalesce(func.sum(EventObservation.max_n), 0),
        )
        .join(Event, Event.deployment_id == Deployment.id)
        .join(EventObservation, EventObservation.event_id == Event.id)
        .join(Project, Project.id == Deployment.project_id)
        .where(*project_filter)
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
