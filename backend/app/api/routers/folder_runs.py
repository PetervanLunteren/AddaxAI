"""
Folder runs API.

A folder run is a presentational shell over the existing Project +
DeploymentQueue pipeline. The endpoints here orchestrate the underlying
rows so the frontend stepper does not have to know about Sites,
Deployments, or the queue infrastructure.

A folder run owns:
- One `Project` row with `mode='folder_run'` and a small JSON state
  blob on `Project.folder_run_state` carrying the current step.
- One `DeploymentQueue` entry with `site_id=NULL` pointing at the
  source folder. The standard queue worker turns this into a
  `Deployment` once analysis runs.

The user-facing deliverables (CSV, recognition JSON, visualised images,
blurred copies, separated subfolders) land in a sibling `AddaxAI
results/<run_name>/` folder; that part of the flow ships in a later
slice and is not orchestrated here yet.

Following DEVELOPERS.md principles: type hints everywhere, crash on
unexpected errors, no silent failures.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.crud import deployment_queue as crud_queue
from app.api.crud import project as crud_project
from app.api.schemas.deployment_queue import (
    DeploymentQueueCreate,
    DeploymentQueueResponse,
)
from app.api.schemas.project import ProjectCreate, ProjectResponse
from app.core.logging_config import get_logger
from app.db.base import get_db

logger = get_logger(__name__)
router = APIRouter(prefix="/api/folder-runs", tags=["Folder runs"])


FolderRunStep = Literal["folder", "model", "run", "review", "save"]


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------


class FolderRunCreate(BaseModel):
    """Request body for POST /api/folder-runs.

    `source_folder` is the absolute path the user picked. `name` is
    optional and defaults to the folder's basename. `video_count` /
    `image_count` come from the same client-side folder scan that
    drives the deployment-queue create flow.
    """

    source_folder: str = Field(..., min_length=1)
    name: str | None = Field(None, min_length=1, max_length=255)
    video_count: int = Field(default=0, ge=0)
    image_count: int = Field(default=0, ge=0)


class FolderRunStepUpdate(BaseModel):
    """Request body for PATCH /api/folder-runs/{id}/step."""

    step: FolderRunStep


class FolderRunResponse(BaseModel):
    """Shape returned by the folder-run endpoints.

    Carries the project the run is wrapped around, the queue entry
    that points at its source folder, and the current step. The
    queue entry is present for resume so the frontend can show
    progress when the user reopens an in-flight run.
    """

    project: ProjectResponse
    queue_entry: DeploymentQueueResponse | None
    step: FolderRunStep

    model_config = {"from_attributes": True}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _auto_name(source_folder: str) -> str:
    """Derive a folder-run name from the chosen folder.

    Falls back to a timestamp when the path is empty or all separator
    (rare; the API validates min_length=1 so this is defensive).
    """
    basename = Path(source_folder).name
    if basename:
        return basename
    return f"folder-run-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


def _load_run(db: Session, run_id: str) -> FolderRunResponse:
    """Load a folder run by project id and assemble the response shape.

    Raises HTTP 404 when the project does not exist or is not a folder
    run, so the frontend cannot stumble into a research project via a
    misclicked URL.
    """
    project = crud_project.get_project(db, run_id)
    if project is None or project.mode != "folder_run":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder run with id '{run_id}' not found",
        )

    # A folder run has at most one queue entry. If there are zero,
    # it means the user landed on the resume URL of a run that was
    # cleaned up; we treat that as "no queue entry yet" and let the
    # frontend handle the empty case.
    entries = crud_queue.get_queue_entries(db, project.id, status=None)
    queue_entry = entries[0] if entries else None

    state: dict = project.folder_run_state or {}
    step: FolderRunStep = state.get("step", "folder")

    return FolderRunResponse(
        project=ProjectResponse.model_validate(project),
        queue_entry=(
            DeploymentQueueResponse.model_validate(queue_entry)
            if queue_entry is not None
            else None
        ),
        step=step,
    )


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


@router.post(
    "", response_model=FolderRunResponse, status_code=status.HTTP_201_CREATED
)
def create_folder_run(
    payload: FolderRunCreate, db: Session = Depends(get_db)
) -> FolderRunResponse:
    """Create a folder run.

    Atomic in spirit: project row + queue entry both written, errors
    bubble up to the unhandled-exception middleware and surface a 500
    to the client. Timezone defaults to UTC because folder runs do not
    expose the sun-overlay / Camtrap-DP flows that depend on it; the
    promotion dialog asks the user for a real timezone when they
    convert a folder run into a research project.
    """
    name = payload.name or _auto_name(payload.source_folder)

    project_create = ProjectCreate(
        name=name,
        timezone="UTC",
        mode="folder_run",
        folder_run_state={
            "step": "folder",
            "source_folder": payload.source_folder,
        },
    )

    try:
        project = crud_project.create_project(db, project_create)
    except Exception as e:
        # Duplicate names get IntegrityError; we surface a 409 the same
        # way the regular project create endpoint does.
        from sqlalchemy.exc import IntegrityError

        if isinstance(e, IntegrityError):
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A folder run named '{name}' already exists",
            ) from e
        raise

    queue_create = DeploymentQueueCreate(
        project_id=project.id,
        folder_path=payload.source_folder,
        site_id=None,
        video_count=payload.video_count,
        image_count=payload.image_count,
    )
    queue_entry = crud_queue.create_queue_entry(db, queue_create)

    logger.info(
        f"Created folder run: project_id={project.id} "
        f"name={name!r} folder={payload.source_folder!r}"
    )
    return FolderRunResponse(
        project=ProjectResponse.model_validate(project),
        queue_entry=DeploymentQueueResponse.model_validate(queue_entry),
        step="folder",
    )


@router.get("/{run_id}", response_model=FolderRunResponse)
def get_folder_run(
    run_id: str, db: Session = Depends(get_db)
) -> FolderRunResponse:
    """Load an existing folder run by project id."""
    return _load_run(db, run_id)


@router.patch("/{run_id}/step", response_model=FolderRunResponse)
def update_folder_run_step(
    run_id: str,
    payload: FolderRunStepUpdate,
    db: Session = Depends(get_db),
) -> FolderRunResponse:
    """Persist the current step so resume drops the user back where
    they were.

    Only `folder_run_state.step` is touched; the rest of the JSON
    blob (source_folder, save options written by later slices) is
    preserved. Validation of the step name happens at the Pydantic
    layer via the `FolderRunStep` Literal.
    """
    project = crud_project.get_project(db, run_id)
    if project is None or project.mode != "folder_run":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder run with id '{run_id}' not found",
        )

    state: dict = dict(project.folder_run_state or {})
    state["step"] = payload.step
    project.folder_run_state = state
    db.commit()
    db.refresh(project)

    return _load_run(db, run_id)
