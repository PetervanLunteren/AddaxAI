"""
Job API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.crud import job as crud_job
from app.api.schemas.job import JobCreate, JobResponse, JobUpdate, RunQueueResponse
from app.core.logging_config import get_logger
from app.db.base import get_db

logger = get_logger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["Jobs"])


@router.get("", response_model=list[JobResponse])
def list_jobs(
    type: str | None = Query(None, description="Filter by job type"),
    status: str | None = Query(None, description="Filter by job status"),
    project_id: str | None = Query(None, description="Filter by project_id in payload"),
    db: Session = Depends(get_db),
) -> list[JobResponse]:
    """
    List all jobs, optionally filtered by type, status, and/or project_id.

    If project_id is provided, filters jobs where payload contains that project_id.
    Returns empty list if no jobs exist.
    """
    if project_id:
        # Use project-specific query
        jobs = crud_job.get_jobs_by_project(db, project_id, job_type=type)
        # Further filter by status if provided
        if status:
            jobs = [j for j in jobs if j.status == status]
    else:
        # Use general query
        jobs = crud_job.get_jobs(db, job_type=type, status=status)

    return [JobResponse.model_validate(j) for j in jobs]


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job(job: JobCreate, db: Session = Depends(get_db)) -> JobResponse:
    """
    Create a new job (add to queue).

    Job starts with status='pending' and will be processed when queue runs.
    """
    db_job = crud_job.create_job(db, job)
    logger.info(f"Created job {db_job.id} (type: {db_job.type})")
    return JobResponse.model_validate(db_job)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    """
    Get job by ID.

    Returns 404 if job doesn't exist.
    """
    db_job = crud_job.get_job(db, job_id)
    if db_job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id '{job_id}' not found",
        )
    return JobResponse.model_validate(db_job)


@router.patch("/{job_id}", response_model=JobResponse)
def update_job(
    job_id: str, job_update: JobUpdate, db: Session = Depends(get_db)
) -> JobResponse:
    """
    Update an existing job.

    Returns 404 if job doesn't exist.
    Used to update status, progress, results, or errors.
    """
    db_job = crud_job.update_job(db, job_id, job_update)
    if db_job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id '{job_id}' not found",
        )
    logger.info(f"Updated job {job_id}: {job_update.model_dump(exclude_unset=True)}")
    return JobResponse.model_validate(db_job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a job (remove from queue).

    Returns 404 if job doesn't exist.
    Only pending jobs should be deleted. Running/completed jobs should be cancelled instead.
    """
    db_job = crud_job.get_job(db, job_id)
    if db_job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id '{job_id}' not found",
        )

    # Warn if trying to delete non-pending job
    if db_job.status != "pending":
        logger.warning(
            f"Deleting job {job_id} with status '{db_job.status}' - "
            f"consider cancelling instead"
        )

    deleted = crud_job.delete_job(db, job_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id '{job_id}' not found",
        )

    logger.info(f"Deleted job {job_id}")


@router.post("/run-queue", response_model=RunQueueResponse)
def run_queue(
    project_id: str | None = Query(None, description="Filter by project_id"),
    db: Session = Depends(get_db)
) -> RunQueueResponse:
    """
    Trigger processing of pending deployment_analysis jobs.

    If project_id is provided, only processes jobs for that project.
    For MVP: This is a no-op that returns success.
    Later: Will trigger AsyncIO workers to process jobs.

    Returns count of jobs that were started.
    """
    # Get pending deployment_analysis jobs
    if project_id:
        # Get project-specific pending jobs
        all_pending = crud_job.get_jobs_by_project(db, project_id, job_type="deployment_analysis")
        pending_jobs = [j for j in all_pending if j.status == "pending"]
    else:
        # Get all pending jobs
        pending_jobs = crud_job.get_pending_jobs(db, job_type="deployment_analysis")

    job_ids = [job.id for job in pending_jobs]
    job_count = len(job_ids)

    # TODO: When ML is implemented, start AsyncIO workers here
    # For now, just log and return success
    logger.info(
        f"Run queue triggered for project {project_id or 'all'}: "
        f"{job_count} pending deployment_analysis jobs found (no-op for MVP)"
    )

    return RunQueueResponse(
        message=f"Queue processing started (MVP: no-op). {job_count} jobs would be processed.",
        jobs_started=job_count,
        job_ids=job_ids,
    )
