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
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.crud import deployment_queue as crud_queue
from app.api.crud import job as job_crud
from app.api.crud import project as crud_project
from app.api.schemas.deployment_queue import (
    DeploymentQueueCreate,
    DeploymentQueueResponse,
)
from app.api.schemas.job import JobCreate
from app.api.schemas.project import ProjectCreate, ProjectResponse
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.postprocessing_outputs.exif_metadata import ExifMode
from app.ml.postprocessing_outputs.output_preview import (
    build_output_preview,
)
from app.ml.postprocessing_outputs.separate_folders import (
    SeparateGroupBy,
    SeparateMode,
)
from app.models import DeploymentQueue, Project

logger = get_logger(__name__)
router = APIRouter(prefix="/api/folder-runs", tags=["Folder runs"])


FolderRunStep = Literal[
    "folder", "model", "run", "review", "overview", "save"
]


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


class SaveOutputsRequest(BaseModel):
    """Request body for POST /api/folder-runs/{id}/save-outputs.

    `output_dir` is the absolute path the deliverables should land in.
    Each boolean flag toggles one output module. Only flags that map
    to a shipped module produce files; others are accepted but report
    "module not yet implemented" so the UI can render a placeholder.
    """

    output_dir: str = Field(..., min_length=1)
    separate_folders: bool = False
    # File placement when separate_folders is on. Default is the
    # safest option; move rewrites File.file_path in the DB so the
    # verify UI keeps working post-move; symlink may fail on Windows
    # without Developer Mode (each failed link is recorded per-file).
    separate_method: SeparateMode = "copy"
    # How animal files are grouped under the separated/ root.
    # `taxonomic` produces a nested Class/Order/Family/Genus/species
    # tree; `flat` produces a single folder per species label.
    separate_group_by: SeparateGroupBy = "taxonomic"
    # Label identifiers to exclude from every output. Each entry is
    # either a LabelTaxonomy.id UUID (for taxonomy-mapped labels) or
    # a raw Detection.label string (for unmapped labels under the
    # tree's "Other" branch). Applied as a file-level filter for
    # Separate / Visualise / Blur / EXIF copies and as a row-level
    # filter for CSV / XLSX / recognition JSON.
    excluded_label_ids: list[str] = Field(default_factory=list)
    visualised_images: bool = False
    blur_people: bool = False
    # Explicit EXIF writer. The other modules already embed detection
    # metadata into the copies they create silently; this flag is the
    # opt-in for writing tags onto the source files in place
    # (`exif_mode="overwrite"`) or producing tagged copies in
    # `<output>/exif-tagged/` (`exif_mode="copy"`).
    write_exif: bool = False
    exif_mode: ExifMode = "copy"
    recognition_json: bool = False
    csv: bool = False
    xlsx: bool = False


class OutputPreviewRequest(BaseModel):
    """Body for POST /api/folder-runs/{id}/output-preview.

    Carries the label exclusion set so the preview reflects the
    user's filter state. Empty list = no filter, every label
    contributes. Each entry is a LabelTaxonomy.id UUID or a raw
    label string (matching the heterogeneous output of the
    label-tree endpoint).
    """

    excluded_label_ids: list[str] = Field(default_factory=list)


class OutputPreviewResponse(BaseModel):
    """Aggregate counts the Save step uses to render a live folder
    preview. Computed deterministically from the project's DB state;
    the numbers are exact, not estimates, because the placement
    rules are fixed.
    """

    total_files: int
    image_count: int
    video_count: int
    total_bytes: int
    files_with_known_size: int
    dropped_by_filter: int
    in_scope_files: int
    in_scope_image_count: int
    in_scope_video_count: int
    in_scope_bytes: int
    by_taxonomic_tree: dict[str, int]
    by_flat: dict[str, int]
    multi_species_files: int


class SaveOutputsResponse(BaseModel):
    """Job-id handle returned when the user kicks off Save outputs.

    The actual per-module results land on the job's WebSocket
    completion event (``data`` payload) — same shape the
    synchronous endpoint used to return inline. The frontend
    subscribes to the job via ``useTaskProgress`` to render the
    blocking progress modal and capture the final result.
    """

    job_id: str


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


def _find_existing_run(db: Session, source_folder: str) -> Project | None:
    """Find a folder-run project already pointing at this source folder.

    Matches legacy AddaxAI: re-selecting an already-analysed folder
    re-opens it instead of starting from scratch. Returns the most
    recently updated match so a user with stale duplicates from
    before this resume logic existed still lands on the latest one.

    Pre-existing duplicates are not deleted here — they stay
    invisible (no list surfaces them) and harmless.
    """
    stmt = (
        select(Project)
        .join(DeploymentQueue, DeploymentQueue.project_id == Project.id)
        .where(Project.mode == "folder_run")
        .where(DeploymentQueue.folder_path == source_folder)
        .order_by(Project.updated_at_utc.desc())
        .limit(1)
    )
    return db.execute(stmt).scalar_one_or_none()


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
    """Create or resume a folder run for a source folder.

    Legacy AddaxAI behaviour: picking a folder that has already been
    analysed re-opens that run instead of starting fresh. The
    frontend has no "recent work" list; revisiting is done by
    pointing at the same folder again. So this endpoint is
    "create-or-resume": if a folder-run project already points at
    `source_folder`, return it as-is (with its persisted step), and
    the caller navigates to that step.

    For new folders the original path applies: create the project +
    queue entry with `step='folder'`. Timezone defaults to UTC
    because folder runs do not expose the sun-overlay / Camtrap-DP
    flows that depend on it; the promotion dialog asks for a real
    timezone when the user converts a folder run into a research
    project.
    """
    existing = _find_existing_run(db, payload.source_folder)
    if existing is not None:
        # Re-submitting the folder picker counts as completing the
        # folder step. Bump the persisted step forward to "model" if
        # it was still on "folder" (covers legacy runs created
        # before this advance landed, plus any future case where
        # we leave the step un-bumped). Don't touch later persisted
        # steps — they encode actual progress the user made.
        state: dict = dict(existing.folder_run_state or {})
        if state.get("step") == "folder":
            state["step"] = "model"
            existing.folder_run_state = state
            db.commit()
            db.refresh(existing)
        logger.info(
            f"Resuming folder run: project_id={existing.id} "
            f"folder={payload.source_folder!r}"
        )
        return _load_run(db, existing.id)

    name = payload.name or _auto_name(payload.source_folder)

    # Persist step="model" because the act of creating the run is
    # what completes the folder picker. The user is about to land on
    # the Setup (model) step, so that's the right resume target if
    # they close the tab now.
    project_create = ProjectCreate(
        name=name,
        timezone="UTC",
        mode="folder_run",
        folder_run_state={
            "step": "model",
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
        step="model",
    )


@router.get("/{run_id}", response_model=FolderRunResponse)
def get_folder_run(
    run_id: str, db: Session = Depends(get_db)
) -> FolderRunResponse:
    """Load an existing folder run by project id."""
    return _load_run(db, run_id)


@router.post(
    "/{run_id}/output-preview", response_model=OutputPreviewResponse
)
def get_output_preview(
    run_id: str,
    payload: OutputPreviewRequest | None = None,
    db: Session = Depends(get_db),
) -> OutputPreviewResponse:
    """Return the file / placement counts the Save step uses to
    render a live folder-tree preview.

    POST so the request body can carry the species exclusion set —
    a long species list as a query string would be awkward. Pure
    read; no side effects.
    """
    project = crud_project.get_project(db, run_id)
    if project is None or project.mode != "folder_run":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder run with id '{run_id}' not found",
        )
    excluded = (
        frozenset(payload.excluded_label_ids)
        if payload and payload.excluded_label_ids
        else None
    )
    preview = build_output_preview(
        db, run_id, excluded_label_ids=excluded
    )
    return OutputPreviewResponse(**preview.to_dict())


@router.post("/{run_id}/save-outputs", response_model=SaveOutputsResponse)
async def save_outputs(
    run_id: str,
    payload: SaveOutputsRequest,
    db: Session = Depends(get_db),
) -> SaveOutputsResponse:
    """Spawn a background job that runs the chosen postprocess outputs.

    Returns ``{"job_id": ...}`` immediately. The frontend subscribes
    to the job's WebSocket channel via ``useTaskProgress`` and renders
    a blocking progress modal. The job's ``result`` payload on
    completion carries the per-module ``to_dict()`` summaries the UI
    used to receive on the synchronous response.
    """
    project = crud_project.get_project(db, run_id)
    if project is None or project.mode != "folder_run":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder run with id '{run_id}' not found",
        )

    output_root = Path(payload.output_dir)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not create output directory: {e}",
        ) from e

    job = job_crud.create_job(
        db,
        JobCreate(
            type="folder_run_save_outputs",
            payload={
                "run_id": run_id,
                "output_dir": str(output_root),
                "separate_folders": payload.separate_folders,
                "separate_method": payload.separate_method,
                "separate_group_by": payload.separate_group_by,
                "visualised_images": payload.visualised_images,
                "blur_people": payload.blur_people,
                "write_exif": payload.write_exif,
                "exif_mode": payload.exif_mode,
                "recognition_json": payload.recognition_json,
                "csv": payload.csv,
                "xlsx": payload.xlsx,
                "excluded_label_ids": list(payload.excluded_label_ids),
            },
        ),
    )

    from app.workers.folder_run_save_outputs_worker import (
        process_save_outputs_job,
    )

    ws_manager.register_start(
        job.id, lambda jid=job.id: process_save_outputs_job(jid)
    )

    logger.info(
        f"save_outputs: spawned job={job.id} run={run_id} "
        f"dir={output_root}"
    )
    return SaveOutputsResponse(job_id=job.id)


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
