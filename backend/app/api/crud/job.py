"""
CRUD operations for Job model.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling (let exceptions bubble up)
- No silent failures
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.job import JobCreate, JobUpdate
from app.models.job import Job


def get_jobs(
    db: Session, job_type: str | None = None, status: str | None = None
) -> list[Job]:
    """
    Get all jobs, optionally filtered by type and/or status.

    Returns empty list if no jobs exist.
    """
    query = select(Job).order_by(Job.created_at_utc.desc())

    if job_type:
        query = query.where(Job.type == job_type)

    if status:
        query = query.where(Job.status == status)

    result = db.execute(query)
    return list(result.scalars().all())


def get_job(db: Session, job_id: str) -> Job | None:
    """
    Get job by ID.

    Returns None if job doesn't exist.
    """
    result = db.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


def create_job(db: Session, job: JobCreate) -> Job:
    """
    Create a new job.

    Job starts with status='pending' and progress_current=0.
    Crashes if database constraint violated.
    """
    db_job = Job(
        type=job.type,
        status="pending",
        progress_current=0,
        progress_total=None,
        payload=job.payload,
        result=None,
        error=None,
        created_at_utc=datetime.now(UTC),
        started_at_utc=None,
        completed_at_utc=None,
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def update_job(db: Session, job_id: str, job_update: JobUpdate) -> Job | None:
    """
    Update an existing job.

    Returns None if job doesn't exist.
    Only updates fields that are provided (not None).
    """
    db_job = get_job(db, job_id)
    if db_job is None:
        return None

    # Update only provided fields
    update_data = job_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_job, field, value)

    # Update timestamps based on status changes
    if "status" in update_data:
        if update_data["status"] == "running" and db_job.started_at_utc is None:
            db_job.started_at_utc = datetime.now(UTC)
        elif update_data["status"] in ("completed", "failed", "cancelled"):
            db_job.completed_at_utc = datetime.now(UTC)

    db.commit()
    db.refresh(db_job)
    return db_job


def delete_job(db: Session, job_id: str) -> bool:
    """
    Delete a job.

    Returns True if deleted, False if job doesn't exist.
    """
    db_job = get_job(db, job_id)
    if db_job is None:
        return False

    db.delete(db_job)
    db.commit()
    return True


def update_job_status(db: Session, job_id: str, status: str) -> Job | None:
    """
    Update job status (convenience function).

    Args:
        db: Database session
        job_id: Job ID
        status: New status ("pending", "running", "completed", "failed", "cancelled")

    Returns:
        Updated Job or None if job doesn't exist
    """
    from app.api.schemas.job import JobUpdate

    return update_job(db, job_id, JobUpdate(status=status))


def reconcile_interrupted_jobs(db: Session) -> int:
    """
    Fail jobs left `running` by a previous process. Call once on startup.

    Analysis runs in an in-memory worker. A backend restart or crash
    kills that worker but leaves the job row `running` forever, so it
    shows as perpetually in progress and its queue entries stay
    `processing`. At startup no job can legitimately be running yet, so
    every `running` job is marked `failed` and every still-`processing`
    queue entry is reset to `failed` too.

    Returns the number of jobs reconciled.
    """
    from app.models.deployment_queue import DeploymentQueue

    interrupted = list(
        db.execute(select(Job).where(Job.status == "running")).scalars().all()
    )
    stuck = list(
        db.execute(
            select(DeploymentQueue).where(DeploymentQueue.status == "processing")
        )
        .scalars()
        .all()
    )
    if not interrupted and not stuck:
        return 0

    now = datetime.now(UTC)
    reason = "Interrupted by a server restart before completion"

    for job in interrupted:
        job.status = "failed"
        job.error = reason
        job.completed_at_utc = now

    for entry in stuck:
        entry.status = "failed"
        if not entry.error:
            entry.error = reason
        entry.processed_at_utc = now

    db.commit()
    return len(interrupted)


def get_pending_jobs(db: Session, job_type: str | None = None) -> list[Job]:
    """
    Get all pending jobs, optionally filtered by type.

    Useful for queue processing.
    """
    return get_jobs(db, job_type=job_type, status="pending")


def get_jobs_by_project(db: Session, project_id: str, job_type: str | None = None) -> list[Job]:
    """
    Get all jobs for a specific project by filtering payload.

    This queries jobs where payload contains the project_id.
    Works for deployment_analysis jobs which have project_id in payload.

    Args:
        db: Database session
        project_id: Project ID to filter by
        job_type: Optional job type filter

    Returns:
        List of jobs matching the project_id in payload
    """
    query = select(Job).order_by(Job.created_at_utc.desc())

    if job_type:
        query = query.where(Job.type == job_type)

    # Execute query and filter by project_id in payload (JSON field)
    result = db.execute(query)
    all_jobs = list(result.scalars().all())

    # Filter by project_id in payload
    project_jobs = [
        job for job in all_jobs
        if job.payload and job.payload.get("project_id") == project_id
    ]

    return project_jobs
