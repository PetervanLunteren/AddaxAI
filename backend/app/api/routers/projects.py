"""
Project API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import project as crud_project
from app.api.schemas.project import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ProjectWithStats,
)
from app.core.logging_config import get_logger
from app.db.base import get_db
from app.models import Detection, Deployment, File, Site

logger = get_logger(__name__)
router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("", response_model=list[ProjectResponse])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectResponse]:
    """
    List all projects.

    Returns empty list if no projects exist.
    """
    projects = crud_project.get_projects(db)
    return [ProjectResponse.model_validate(p) for p in projects]


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    project: ProjectCreate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """
    Create a new project.

    Validates that selected models exist before creating project.
    Normalizes "none" to NULL for classification_model_id.

    Returns 400 if model IDs are invalid.
    Returns 409 if project name already exists.
    """
    from app.ml.manifest_manager import ManifestManager
    from app.core.config import get_settings

    # Validate models exist
    settings = get_settings()
    manifest_mgr = ManifestManager(settings.user_data_dir / "models")

    # Validate detection model
    try:
        manifest_mgr.get_model(project.detection_model_id)
    except ValueError:
        logger.warning(f"Invalid detection model: {project.detection_model_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Detection model '{project.detection_model_id}' not found",
        )

    # Normalize "none" to NULL and validate classification model
    if project.classification_model_id == "none":
        project.classification_model_id = None
    elif project.classification_model_id is not None:
        try:
            manifest_mgr.get_model(project.classification_model_id)
        except ValueError:
            logger.warning(f"Invalid classification model: {project.classification_model_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Classification model '{project.classification_model_id}' not found",
            )

    try:
        db_project = crud_project.create_project(db, project)
        logger.info(f"Created project: {project.name} (ID: {db_project.id})")
        return ProjectResponse.model_validate(db_project)
    except IntegrityError as e:
        logger.warning(f"Failed to create project '{project.name}': duplicate name")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project with name '{project.name}' already exists",
        ) from e


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    """
    Get project by ID.

    Returns 404 if project doesn't exist.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        logger.warning(f"Project not found: {project_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )
    return ProjectResponse.model_validate(db_project)


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str, project: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """
    Update an existing project.

    Returns 404 if project doesn't exist.
    Returns 409 if new name conflicts with existing project.
    """
    try:
        db_project = crud_project.update_project(db, project_id, project)
        if db_project is None:
            logger.warning(f"Cannot update project: {project_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with id '{project_id}' not found",
            )
        logger.info(f"Updated project: {project_id}")
        return ProjectResponse.model_validate(db_project)
    except IntegrityError as e:
        logger.warning(f"Failed to update project {project_id}: duplicate name")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project name already exists",
        ) from e


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a project.

    Returns 404 if project doesn't exist.
    Cascades deletion to all sites, deployments, files, etc.
    """
    deleted = crud_project.delete_project(db, project_id)
    if not deleted:
        logger.warning(f"Cannot delete project: {project_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )
    logger.info(f"Deleted project: {project_id} (cascaded to all related data)")


@router.get("/{project_id}/stats", response_model=ProjectWithStats)
def get_project_stats(
    project_id: str, db: Session = Depends(get_db)
) -> ProjectWithStats:
    """
    Get project with statistics.

    Returns project info plus counts of sites, deployments, files, and detections.
    Returns 404 if project doesn't exist.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    stats = crud_project.get_project_stats(db, project_id)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    # Combine project data with stats
    project_dict = ProjectResponse.model_validate(db_project).model_dump()
    project_dict.update(stats)

    return ProjectWithStats(**project_dict)


@router.get("/{project_id}/detection-stats")
def get_detection_stats(project_id: str, db: Session = Depends(get_db)) -> dict:
    """
    Get detection category statistics for a project.

    Returns counts by category (animal, person, vehicle).
    """
    # Query detection counts grouped by category
    stats = (
        db.query(Detection.category, func.count(Detection.id).label("count"))
        .join(File)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .group_by(Detection.category)
        .all()
    )

    # Convert to dict
    result = {category: count for category, count in stats}

    return result


@router.get("/{project_id}/species-stats")
def get_species_stats(project_id: str, db: Session = Depends(get_db)) -> list[dict]:
    """
    Get top species counts for a project.

    Returns list of {species, count} sorted by count descending, top 10.
    Only includes detections with a species classification.
    """
    stats = (
        db.query(Detection.species, func.count(Detection.id).label("count"))
        .join(File)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .filter(Detection.species.isnot(None))
        .group_by(Detection.species)
        .order_by(func.count(Detection.id).desc())
        .all()
    )

    return [{"species": species, "count": count} for species, count in stats]


@router.post(
    "/{project_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_classifications(
    project_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Reprocess classifications for a project.

    Launches an async task that sends WebSocket progress updates.
    Returns immediately with a job_id for progress tracking.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    from app.api.crud import job as crud_job
    from app.api.schemas.job import JobCreate

    job_data = JobCreate(
        type="postprocessing",
        payload={"project_id": project_id},
    )
    job = crud_job.create_job(db, job_data)
    logger.info(f"Created postprocessing job {job.id} for project {project_id}")

    from app.workers.postprocessing_worker import process_postprocessing_job

    asyncio.create_task(process_postprocessing_job(job.id))

    return {"message": "Postprocessing started", "job_id": job.id}


@router.get("/{project_id}/postprocessing-status")
def get_postprocessing_status(
    project_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Check whether postprocessing needs to be re-run.

    Compares current smoothing settings hash with stored hash.

    Returns:
        {needs_reprocessing: bool, has_classifications: bool}
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    # Check if any classifications exist for this project
    has_cls = (
        db.query(Detection.id)
        .join(File)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .filter(Detection.species.isnot(None))
        .limit(1)
        .first()
    ) is not None

    # Compute current hash and compare
    from app.ml.postprocessing import compute_postprocessing_settings_hash

    current_hash = compute_postprocessing_settings_hash(db_project)
    stored_hash = db_project.postprocessing_settings_hash

    needs_reprocessing = has_cls and (current_hash != stored_hash)

    return {
        "needs_reprocessing": needs_reprocessing,
        "has_classifications": has_cls,
    }
