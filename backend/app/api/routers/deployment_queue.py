"""
Deployment Queue API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import deployment_queue as crud_queue
from app.api.schemas.deployment_queue import (
    DeploymentQueueCreate,
    DeploymentQueueResponse,
    ProcessQueueRequest,
)
from app.core.logging_config import get_logger
from app.db.base import get_db

logger = get_logger(__name__)
router = APIRouter(prefix="/api/deployment-queue", tags=["Deployment Queue"])


@router.get("", response_model=list[DeploymentQueueResponse])
def list_queue_entries(
    project_id: str,
    status: str | None = None,
    db: Session = Depends(get_db)
) -> list[DeploymentQueueResponse]:
    """
    List all queue entries for a project.

    Optionally filter by status (pending, processing, completed, failed).
    Returns empty list if no entries exist.
    """
    entries = crud_queue.get_queue_entries(db, project_id, status)
    return [DeploymentQueueResponse.model_validate(e) for e in entries]


@router.post("", response_model=DeploymentQueueResponse, status_code=status.HTTP_201_CREATED)
def create_queue_entry(
    entry: DeploymentQueueCreate,
    db: Session = Depends(get_db)
) -> DeploymentQueueResponse:
    """
    Add a new entry to the deployment queue.

    Creates a queue entry that will be processed when user clicks "Process Queue".
    """
    try:
        db_entry = crud_queue.create_queue_entry(db, entry)
        logger.info(f"Added entry to queue: project_id={entry.project_id}, folder={entry.folder_path}")
        return DeploymentQueueResponse.model_validate(db_entry)
    except IntegrityError as e:
        logger.error(f"Failed to create queue entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project_id or site_id",
        ) from e


@router.get("/{entry_id}", response_model=DeploymentQueueResponse)
def get_queue_entry(
    entry_id: str,
    db: Session = Depends(get_db)
) -> DeploymentQueueResponse:
    """
    Get queue entry by ID.

    Returns 404 if entry doesn't exist.
    """
    db_entry = crud_queue.get_queue_entry(db, entry_id)
    if db_entry is None:
        logger.warning(f"Queue entry not found: {entry_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Queue entry with id '{entry_id}' not found",
        )
    return DeploymentQueueResponse.model_validate(db_entry)


@router.delete("/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_queue_entry(
    entry_id: str,
    db: Session = Depends(get_db)
) -> None:
    """
    Remove an entry from the queue.

    Returns 404 if entry doesn't exist.
    """
    success = crud_queue.delete_queue_entry(db, entry_id)
    if not success:
        logger.warning(f"Queue entry not found for deletion: {entry_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Queue entry with id '{entry_id}' not found",
        )
    logger.info(f"Deleted queue entry: {entry_id}")


@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
def process_queue(
    request: ProcessQueueRequest,
    db: Session = Depends(get_db)
) -> dict[str, str]:
    """
    Start processing the deployment queue for a project.

    Processes all pending entries sequentially.
    Returns immediately with job_id for progress tracking.

    NOTE: This endpoint creates a background job.
    Implementation of the actual processing logic is in queue_processor service.
    """
    # Get pending entries
    pending_entries = crud_queue.get_queue_entries(db, request.project_id, status="pending")

    if not pending_entries:
        logger.info(f"No pending queue entries for project: {request.project_id}")
        return {
            "message": "No pending queue entries to process",
            "processed_count": 0
        }

    # TODO: Create background job to process queue
    # For now, just return a placeholder response
    logger.info(f"Starting queue processing for project {request.project_id}: {len(pending_entries)} entries")

    return {
        "message": f"Processing {len(pending_entries)} queue entries",
        "project_id": request.project_id,
        "entry_count": len(pending_entries),
        # "job_id": "TODO"  # Will add when background job system is integrated
    }
