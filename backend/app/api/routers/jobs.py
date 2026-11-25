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
from app.api.schemas.job import JobResponse
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
