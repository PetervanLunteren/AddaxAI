"""
Deployment Queue API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import deployment_queue as crud_queue
from app.api.crud import job as crud_job
from app.api.schemas.csv_import import (
    CsvImportProblem,
    CsvImportResult,
    DeploymentImportPreview,
    DeploymentImportRow,
)
from app.api.schemas.deployment_queue import (
    DeploymentQueueCreate,
    DeploymentQueueResponse,
    ProcessQueueRequest,
)
from app.api.schemas.job import JobCreate
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.models import Project
from app.services.csv_import import MAX_CSV_BYTES, drop_problem_rows
from app.services.csv_import_deployments import (
    CAMERA_OFFSETS_NEED_PAIRED,
    check_paired_camera_layout,
    normalize_folder,
    parse_deployment_csv,
    resolve_site_ids,
    validate_deployment_rows,
)
from app.workers import process_deployment_analysis

logger = get_logger(__name__)
router = APIRouter(prefix="/api/deployment-queue", tags=["Deployment Queue"])


def _require_project(db: Session, project_id: str) -> None:
    """404 when the project is gone, with the wording the other routers use."""
    if db.get(Project, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )


def _read_csv_upload(file: UploadFile) -> bytes:
    """The uploaded bytes, refusing anything too big to hold in memory."""
    contents = file.file.read()
    if len(contents) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file is larger than 2 MB. Import your deployments in smaller files.",
        )
    return contents


def _check_deployment_csv(
    db: Session, project_id: str, contents: bytes
) -> tuple[list[DeploymentImportRow], list[CsvImportProblem]]:
    """Parse and validate in one go, the way both import routes need it.

    Rows that turned out to have a problem are dropped, so what comes back is
    exactly what would be queued, with its media counts filled in.
    """
    rows, problems = parse_deployment_csv(contents)
    rows, validation_problems = validate_deployment_rows(db, project_id, rows)
    problems += validation_problems
    # File-level problems (no row number) first, then in file order.
    problems.sort(key=lambda p: (p.row is not None, p.row or 0))
    return drop_problem_rows(rows, problems), problems


@router.get("", response_model=list[DeploymentQueueResponse])
def list_queue_entries(
    project_id: str, status: str | None = None, db: Session = Depends(get_db)
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
    entry: DeploymentQueueCreate, db: Session = Depends(get_db)
) -> DeploymentQueueResponse:
    """
    Add a new entry to the deployment queue.

    Creates a queue entry that will be processed when user clicks "Process Queue".
    """
    if entry.camera_offsets and not entry.paired_cameras:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=CAMERA_OFFSETS_NEED_PAIRED,
        )
    if entry.paired_cameras:
        try:
            layout_problem = check_paired_camera_layout(entry.folder_path)
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read folder: {e}",
            ) from e
        if layout_problem is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=layout_problem
            )
    try:
        db_entry = crud_queue.create_queue_entry(db, entry)
        logger.info(
            f"Added entry to queue: project_id={entry.project_id}, folder={entry.folder_path}"
        )
        return DeploymentQueueResponse.model_validate(db_entry)
    except IntegrityError as e:
        logger.error(f"Failed to create queue entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project_id or site_id",
        ) from e


@router.post("/import/preview", response_model=DeploymentImportPreview)
def preview_deployment_import(
    file: UploadFile,
    project_id: str = Query(..., description="Project the deployments would be queued in"),
    db: Session = Depends(get_db),
) -> DeploymentImportPreview:
    """
    Check a deployment CSV without writing anything.

    Always 200: per-row problems are the expected case and are reported in
    the body, not raised. An empty `problems` list means the same file can be
    posted to /import.
    """
    _require_project(db, project_id)
    rows, problems = _check_deployment_csv(db, project_id, _read_csv_upload(file))
    return DeploymentImportPreview(rows=rows, problems=problems)


@router.post("/import", response_model=CsvImportResult)
def import_deployments(
    file: UploadFile,
    project_id: str = Query(..., description="Project the deployments are queued in"),
    db: Session = Depends(get_db),
) -> CsvImportResult:
    """
    Import a deployment CSV into the queue, all or nothing.

    The file is checked again rather than trusting the preview: a drive can
    be unplugged or a folder queued in between. Any problem means nothing is
    written and `imported` is 0.

    Entries land as `pending`, exactly as if they had been added one at a
    time on the Process page. Nothing is analysed until the queue is run.
    """
    _require_project(db, project_id)
    rows, problems = _check_deployment_csv(db, project_id, _read_csv_upload(file))
    if problems:
        return CsvImportResult(imported=0, problems=problems)

    site_ids = resolve_site_ids(db, project_id, rows)
    creates = [
        DeploymentQueueCreate(
            project_id=project_id,
            folder_path=normalize_folder(row.folder),
            site_id=site_ids[row.row],
            image_count=row.image_count,
            video_count=row.video_count,
            notes=row.notes,
            paired_cameras=row.paired_cameras,
            tags=row.tags,
        )
        for row in rows
    ]

    try:
        created = crud_queue.create_queue_entries_bulk(db, creates)
    except IntegrityError as e:
        # Unreachable unless the project or a site was deleted between the
        # check above and the insert. Roll back or the session stays unusable.
        db.rollback()
        logger.warning(f"Deployment CSV import failed for project {project_id}: {e}")
        return CsvImportResult(
            imported=0,
            problems=[
                CsvImportProblem(
                    message=(
                        "The deployments could not be saved because the project "
                        "changed during the import. Import the file again."
                    )
                )
            ],
        )

    logger.info(f"Imported {len(created)} queue entries into project {project_id}")
    return CsvImportResult(imported=len(created), problems=[])


@router.get("/{entry_id}", response_model=DeploymentQueueResponse)
def get_queue_entry(entry_id: str, db: Session = Depends(get_db)) -> DeploymentQueueResponse:
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
def delete_queue_entry(entry_id: str, db: Session = Depends(get_db)) -> None:
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
async def process_queue(
    request: ProcessQueueRequest, db: Session = Depends(get_db)
) -> dict[str, str | int | list[str]]:
    """
    Start processing the deployment queue for a project.

    Creates ONE job that will process all queue entries sequentially.
    Returns immediately with the job ID for progress tracking.
    """
    # Get pending entries
    pending_entries = crud_queue.get_queue_entries(db, request.project_id, status="pending")

    if not pending_entries:
        logger.info(f"No pending queue entries for project: {request.project_id}")
        return {
            "message": "No pending queue entries to process",
            "jobs_started": 0,
            "job_ids": [],
            "queue_entry_ids": [],
        }

    entry_count = len(pending_entries)
    logger.info(
        f"Starting sequential queue processing for "
        f"project {request.project_id}: {entry_count} entries"
    )

    # Create ONE job for ALL queue entries
    entry_ids = [entry.id for entry in pending_entries]
    job_create = JobCreate(
        type="deployment_analysis",
        payload={
            "project_id": request.project_id,
            "queue_entry_ids": entry_ids,  # List of ALL entries to process sequentially
            "is_batch_job": True,
        },
    )
    job = crud_job.create_job(db, job_create)

    # Mark ALL entries as processing
    for entry in pending_entries:
        crud_queue.update_queue_status(db, entry.id, status="processing")

    # Register worker to start when frontend sends "ready" over WebSocket
    ws_manager.register_start(job.id, lambda jid=job.id: process_deployment_analysis(jid))

    logger.info(f"Registered batch job {job.id} for {len(entry_ids)} entries")

    return {
        "message": (
            f"Queue processing started. "
            f"{len(entry_ids)} deployments will be processed sequentially."
        ),
        "jobs_started": 1,
        "job_ids": [job.id],
        "queue_entry_ids": entry_ids,
    }
