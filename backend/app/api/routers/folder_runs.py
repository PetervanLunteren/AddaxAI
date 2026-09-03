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

The user-facing deliverables land in the user's output dir, which
defaults to the source folder itself: loose ``addaxai-*`` data files
(CSV, XLSX, recognition JSON, summary) at its root, media copies
(separated subfolders, visualised / blurred images) under the
``addaxai-media`` subfolder.

Following DEVELOPERS.md principles: type hints everywhere, crash on
unexpected errors, no silent failures.
"""

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, case, distinct, func, select
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
from app.core.confidence import DEFAULT_COUNTING_THRESHOLD
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml import detection_checkpoint as ckpt
from app.ml.label_exclusion import threshold_or_verified
from app.ml.postprocessing_outputs.output_preview import (
    build_output_preview,
)
from app.ml.postprocessing_outputs.separate_folders import (
    SeparateGroupBy,
)
from app.models import (
    Deployment,
    DeploymentQueue,
    Detection,
    Event,
    File,
    Project,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/folder-runs", tags=["Folder runs"])


FolderRunStep = Literal["setup", "labels", "save"]

# No cap on the runs returned: the "keep newest per folder" dedupe has to see
# every run to know which one is newest, so the endpoint scans the whole set
# either way. A limit here would only shorten the JSON, at the price of hiding
# runs the user can then neither open nor delete. The step-1 list shows the
# most recent handful and reveals the rest on demand, which keeps that choice
# in the UI where it belongs.

# Forward-compat: map retired step slugs so runs persisted under the old
# names re-attach without failing FolderRunStep validation. The counts
# and summary steps were removed (folder run = run AI without
# ecological interpretation; counts live in projects mode), so runs
# parked there resume on labels, the step right before save. Older
# renames ("observations", "model", "overview") chain to the same
# targets.
_LEGACY_STEP_MAP: dict[str, str] = {
    "model": "setup",
    "observations": "labels",
    "counts": "labels",
    "overview": "labels",
    "summary": "labels",
}


def _normalize_step(raw_step: str) -> "FolderRunStep":
    return _LEGACY_STEP_MAP.get(raw_step, raw_step)  # type: ignore[return-value]


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------


class FolderRunCreate(BaseModel):
    """Request body for POST /api/folder-runs.

    `source_folder` is the absolute path the user picked. `name` is
    optional and defaults to the folder's basename. `video_count` /
    `image_count` come from the same client-side folder scan that
    drives the deployment-queue create flow.

    ``force_new`` is the "Discard and start over" path from the
    folder-picker step: when set and an existing folder-run project
    already points at ``source_folder``, the existing project is
    cascade-deleted (DB rows + on-disk ``.addaxai`` cache) before the
    fresh one is created. Default ``False`` keeps the legacy
    create-or-resume behaviour: an existing run is returned as-is.

    ``use_file_mtime_fallback`` is the opt-in the user ticked in the
    folder scan when nothing in the folder carried a capture date. It
    matters here even though a folder run draws no charts: the exported
    files table carries a ``datetime`` column and the run README reports
    the capture range, so without it both come out empty.

    ``datetime_offset_seconds`` is the camera clock correction from the
    Adjust dates modal on the setup step, applied to every capture
    timestamp at ingest. Same plumbing as the deployment queue flow:
    the value is stamped onto the queue entry below.
    """

    source_folder: str = Field(..., min_length=1)
    name: str | None = Field(None, min_length=1, max_length=255)
    video_count: int = Field(default=0, ge=0)
    image_count: int = Field(default=0, ge=0)
    force_new: bool = False
    use_file_mtime_fallback: bool = False
    datetime_offset_seconds: int | None = None


class FolderRunStepUpdate(BaseModel):
    """Request body for PATCH /api/folder-runs/{id}/step."""

    step: FolderRunStep


class FolderRunSummary(BaseModel):
    """One row in the step-1 "Show recent runs" list.

    Deliberately thin: enough to recognise a run (where it was, how big,
    when, how far the review got) and to decide whether it can be resumed.
    The step is not carried: resuming goes through ``GET /{run_id}``, which
    reads the persisted step itself.

    ``detection_count`` / ``verified_detection_count`` are the two halves of
    the "labels verified" fraction the row shows, matching the metric the
    dashboard and the re-run dialog use. ``detection_count`` is 0 for a run
    whose analysis has not produced anything yet, which the UI renders as
    "not analysed yet" rather than a meaningless 0%.

    ``folder_exists`` is false when the source folder has moved, been
    deleted, or lives on a drive that isn't plugged in: the UI greys those
    out rather than letting the user resume into missing files.

    ``queue_status`` is the run's queue entry status, so the row can say
    "previous run did not finish" for a ``failed`` run instead of "not
    analysed yet". Both have zero files, but they are different facts: a
    run killed by a crash or a power cut needs to be run again, and the
    setup step already says so; the list must not contradict it.
    """

    id: str
    folder_path: str
    updated_at_utc: datetime
    file_count: int
    detection_count: int
    verified_detection_count: int
    folder_exists: bool
    queue_status: str


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
    # Id of the analysis job currently running for this run, if any.
    # Lets the frontend re-attach to the progress modal after a refresh
    # without an extra round trip. None when nothing is processing.
    active_job_id: str | None = None

    model_config = {"from_attributes": True}


class SaveOutputsRequest(BaseModel):
    """Request body for POST /api/folder-runs/{id}/save-outputs.

    ``output_dir`` is the absolute path the deliverables should land
    in; it defaults to the source folder itself on the frontend. The
    data exports drop their ``addaxai-*`` files at its root; media
    copies (separation, annotated / blurred images) go under the
    ``addaxai-media`` subfolder. Each boolean flag toggles one output
    module.

    ``draw_bboxes`` and ``anonymise`` drive the combined per-file
    annotated-copies pass: either, neither, or both — when both are
    on, one image per source is written with blurred people/vehicles
    and detection boxes drawn on top.
    """

    output_dir: str = Field(..., min_length=1)
    # Media-output confidence: detections below it (unless verified) are
    # left out of the separated copies, drawn boxes, blurs, and EXIF
    # tags. Data exports (CSV / XLSX / recognition JSON) ignore it: they
    # are always the complete record of the run.
    media_threshold: float = Field(
        DEFAULT_COUNTING_THRESHOLD, ge=0.0, le=1.0
    )
    separate_folders: bool = False
    # How media copies are grouped at the output root. ``taxonomic``
    # nests Class/Order/Family/Genus/species; ``flat`` is one folder
    # per species label; ``none`` copies everything flat at the root.
    separate_group_by: SeparateGroupBy = "flat"
    # Keep a burst together: every file in an event lands in one folder
    # (the event's main species) instead of being filed per file.
    group_events: bool = True
    # Flip the layering to ``<source subfolder>/<species>/`` instead of
    # ``<species>/<source subfolder>/``. Keeps the user's original folders
    # on top and species inside them (the layout camtrapR expects). No
    # effect with ``separate_group_by="none"`` or a flat source folder.
    separate_species_last: bool = False
    # Copy empty captures (no animal / person / vehicle) too. Off by
    # default so the media copies aren't padded with blank captures.
    include_empty: bool = False
    # Label identifiers to exclude from every output. Each entry is
    # either a LabelTaxonomy.id UUID (for taxonomy-mapped labels) or a
    # raw Detection.label string (for unmapped labels under the tree's
    # "Other" branch). Applied as a file-level filter for separate /
    # annotated_copies and as a row-level filter for CSV / XLSX /
    # recognition JSON.
    excluded_label_ids: list[str] = Field(default_factory=list)
    # Draw detection bounding boxes + pill labels on the annotated
    # copies.
    draw_bboxes: bool = False
    # Blur person / vehicle detections on the annotated copies — for
    # sharing datasets without identifying bystanders or vehicles.
    anonymise: bool = False
    recognition_json: bool = False
    csv: bool = False
    xlsx: bool = False
    # Write the ``addaxai-run-info.txt`` manifest. The Save step's "Run
    # details" checkbox. Defaults to True for a client that omits it,
    # which is what the write used to be unconditionally.
    #
    # This field was missing until 2026-08-21, and its absence is the
    # whole of the bug it fixes: the frontend has sent ``run_readme``
    # since 2026-07-14, pydantic ignores fields a model does not declare,
    # so ``false`` was dropped here without a word and the worker's
    # ``payload.get("run_readme", True)`` fell back to writing it. The
    # checkbox never did anything from the day it shipped.
    run_readme: bool = True
    # Which species name to burn into the visualised images: the common
    # name or the scientific name. Mirrors the UI display preference so
    # the saved images match what the user sees. EXIF metadata always
    # carries both names regardless of this choice.
    name_mode: Literal["common", "scientific"] = "common"


class OutputPreviewRequest(BaseModel):
    """Body for POST /api/folder-runs/{id}/output-preview.

    Carries the label exclusion set so the preview reflects the
    user's filter state. Empty list = no filter, every label
    contributes. Each entry is a LabelTaxonomy.id UUID or a raw
    label string (matching the heterogeneous output of the
    label-tree endpoint).
    """

    excluded_label_ids: list[str] = Field(default_factory=list)
    # Media-output confidence, mirroring the save request so the
    # preview counts match what the save will write.
    media_threshold: float = Field(
        DEFAULT_COUNTING_THRESHOLD, ge=0.0, le=1.0
    )
    # Copy empty captures too; off by default. Mirrors the save
    # request so the preview matches what will be written.
    include_empty: bool = False
    # Common vs scientific species-name leaf, mirroring the save request
    # so the previewed tree matches the folders that will be written.
    name_mode: Literal["common", "scientific"] = "common"
    # Event grouping, mirroring the save request so the previewed counts
    # match exactly what the save will write.
    group_events: bool = True
    # Folder layout, mirroring the save request so the previewed tree
    # shows the real on-disk nesting (species folder + preserved source
    # subfolders) in the chosen order.
    separate_group_by: SeparateGroupBy = "flat"
    separate_species_last: bool = False
    # Blur, mirroring the save request: with blur on a video is written
    # as its blurred still instead of the container, so the preview's
    # byte total and filename sample have to follow.
    anonymise: bool = False


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
    by_media_tree: dict[str, int]
    root_files: list[str]


class DetectionResume(BaseModel):
    """What an interrupted run's detection checkpoint holds, for the
    Continue / Start over choice in the re-run dialog. ``images_done ==
    images_total`` means detection finished and only the later phases
    are left to redo."""

    images_done: int
    images_total: int


class FolderRunLookupResponse(BaseModel):
    """Summary the Step 1 folder picker uses to render the
    "already analysed" notice card.

    The numbers are intentionally cheap to compute — a handful of
    aggregate queries that run on every folder-picker change.

    ``verified_detection_count`` is the canonical "how much has the
    user reviewed" number. Signing a File off verifies every visible
    detection on it and rejects the invisible weak ones (see
    ``crud/file.py:set_file_verified``), so this count grows whether
    the user verifies file-by-file or detection-by-detection in the
    verify grid; the rejected boxes stay out of it through the shared
    scope rule.

    Model name fields are resolved through the local manifest; when
    the model is not installed (catalog drift, fresh install) we
    surface the raw id so the card still says something useful.
    """

    id: str
    name: str
    created_at_utc: datetime
    updated_at_utc: datetime
    detection_model_id: str | None
    classification_model_id: str | None
    detection_model_name: str | None
    classification_model_name: str | None
    # The saved detection settings, so the re-run dialog can tell whether
    # the form still matches what ``detection_resume`` was measured under.
    detection_image_size: int | None
    detection_augment: bool
    step: FolderRunStep
    file_count: int
    detection_count: int
    species_count: int
    verified_file_count: int
    verified_detection_count: int
    # Count-confirmation progress (events confirmed on the Counts page),
    # the second half of the app's "Labels verified" + "Counts confirmed"
    # split. ``event_count`` is the denominator for the confirmed percentage.
    event_count: int
    confirmed_event_count: int
    # How far image detection got before the run was interrupted, when
    # there is something to continue from. None for every other run.
    detection_resume: DetectionResume | None = None


class RerunRequest(BaseModel):
    """Body of ``POST /api/folder-runs/{id}/rerun``. Optional: an empty
    body is a plain re-run that clears everything."""

    # Keep the interrupted run's detection checkpoint so the next run
    # continues where detection stopped (the user chose Continue).
    keep_checkpoint: bool = False


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


def _friendly_model_name(model_id: str | None) -> str | None:
    """Resolve a model id to its user-facing friendly name.

    Falls back to the raw id when the manifest does not know the model
    (catalog drift, fresh install, etc.) so the notice card still
    surfaces something the user recognises.
    """
    if not model_id:
        return None
    from app.core.config import get_settings
    from app.ml.manifest_manager import ManifestManager

    settings = get_settings()
    try:
        manifest = ManifestManager(
            settings.models_dir
        ).get_model(model_id)
        return manifest.friendly_name or model_id
    except Exception:
        # Manifest missing, catalog drift, IO error — surface the id.
        return model_id


def _auto_name(source_folder: str) -> str:
    """Derive a folder-run name from the chosen folder.

    Falls back to a timestamp when the path is empty or all separator
    (rare; the API validates min_length=1 so this is defensive).
    """
    basename = Path(source_folder).name
    if basename:
        return basename
    return f"folder-run-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


def _unique_project_name(db: Session, base: str) -> str:
    """Return a Project.name that doesn't already exist in the DB.

    Returns ``base`` if free; otherwise appends `` (2)``, `` (3)``, ...
    until a free name turns up. The bound at 1000 is purely defensive;
    we should never get anywhere near it.

    Folder-run auto-names derive from the source folder basename, so a
    folder previously analysed and then promoted to a research project
    leaves its name occupied. Without this dedup the user re-picking
    the same folder hits a 409. The promote flow can rename later in
    its own dialog if the user wants the cleaner name back.
    """
    candidate = base
    counter = 2
    while db.scalar(select(Project.id).where(Project.name == candidate)):
        candidate = f"{base} ({counter})"
        counter += 1
        if counter > 1000:
            raise RuntimeError(
                f"could not find a free project name starting from {base!r}"
            )
    return candidate


def _find_existing_run(db: Session, source_folder: str) -> Project | None:
    """Find a folder-run project already pointing at this source folder.

    Matches legacy AddaxAI: re-selecting an already-analysed folder
    re-opens it instead of starting from scratch. Returns the most
    recently updated match so a user with stale duplicates from
    before this resume logic existed still lands on the latest one.

    Pre-existing duplicates are not deleted here — they stay
    invisible (no list surfaces them) and harmless.

    Compared in the normalised form the queue stores, so a typed
    trailing slash still finds the run.
    """
    from app.services.csv_import_deployments import normalize_folder

    stmt = (
        select(Project)
        .join(DeploymentQueue, DeploymentQueue.project_id == Project.id)
        .where(Project.mode == "folder_run")
        .where(DeploymentQueue.folder_path == normalize_folder(source_folder))
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
    step: FolderRunStep = _normalize_step(state.get("step", "setup"))

    # Surface the running deployment_analysis job (if any) so the
    # frontend can re-attach to the progress modal after a refresh.
    # The KISS lifecycle is "modal == running run" — without this we
    # could not honour that across reloads.
    active_job = next(
        (
            j
            for j in job_crud.get_jobs_by_project(
                db, project.id, "deployment_analysis"
            )
            if j.status == "running"
        ),
        None,
    )

    return FolderRunResponse(
        project=ProjectResponse.model_validate(project),
        queue_entry=(
            DeploymentQueueResponse.model_validate(queue_entry)
            if queue_entry is not None
            else None
        ),
        step=step,
        active_job_id=active_job.id if active_job is not None else None,
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
    queue entry with `step='setup'`. Timezone defaults to UTC
    because folder runs do not expose the sun-overlay / Camtrap-DP
    flows that depend on it; the promotion dialog asks for a real
    timezone when the user converts a folder run into a research
    project.
    """
    existing = _find_existing_run(db, payload.source_folder)
    if existing is not None and not payload.force_new:
        logger.info(
            f"Resuming folder run: project_id={existing.id} "
            f"folder={payload.source_folder!r}"
        )
        return _load_run(db, existing.id)

    if existing is not None and payload.force_new:
        # "Discard and start over": cascade-delete the existing run +
        # its on-disk .addaxai cache before creating fresh. The
        # destructive confirm dialog on the frontend gates this path.
        logger.info(
            f"Discarding existing folder run: project_id={existing.id} "
            f"folder={payload.source_folder!r}"
        )
        crud_project.delete_folder_run(db, existing.id)

    # Auto-named runs dedup transparently against any colliding project
    # in the DB (including research projects promoted out of an earlier
    # folder run for the same folder). Explicit names from the caller
    # are taken as-is and surface a 409 on collision so the user can
    # rename intentionally.
    if payload.name:
        name = payload.name
    else:
        name = _unique_project_name(db, _auto_name(payload.source_folder))

    # Persist step="setup" because the act of creating the run is
    # what completes the folder picker. The user is about to land on
    # the Setup step, so that's the right resume target if they close
    # the tab now.
    #
    # counting_threshold is the one in-app interpretation floor, same
    # as projects mode: every read path (labels grid default, label
    # tree, lookup summary) and every verification pill measures over
    # it, so they always agree. Storage is unaffected (MegaDetector runs
    # at its 0.01 output cap, everything >= 0.01 is stored) and the exports
    # stay complete (they bypass the threshold). The classification gate
    # is a separate inference knob and no longer pinned to this.
    project_create = ProjectCreate(
        name=name,
        timezone="UTC",
        mode="folder_run",
        counting_threshold=DEFAULT_COUNTING_THRESHOLD,
        folder_run_state={
            "step": "setup",
            "source_folder": payload.source_folder,
        },
    )

    try:
        project = crud_project.create_project(db, project_create)
    except Exception as e:
        # Duplicate names get IntegrityError; surface a 409. The
        # auto-named path is dedup'd above, so this only fires when
        # the caller passed an explicit colliding name.
        from sqlalchemy.exc import IntegrityError

        if isinstance(e, IntegrityError):
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A project named '{name}' already exists",
            ) from e
        raise

    queue_create = DeploymentQueueCreate(
        project_id=project.id,
        folder_path=payload.source_folder,
        site_id=None,
        video_count=payload.video_count,
        image_count=payload.image_count,
        use_file_mtime_fallback=payload.use_file_mtime_fallback,
        datetime_offset_seconds=payload.datetime_offset_seconds,
    )
    queue_entry = crud_queue.create_queue_entry(db, queue_create)

    logger.info(
        f"Created folder run: project_id={project.id} "
        f"name={name!r} folder={payload.source_folder!r}"
    )
    return FolderRunResponse(
        project=ProjectResponse.model_validate(project),
        queue_entry=DeploymentQueueResponse.model_validate(queue_entry),
        step="setup",
    )


@router.get("", response_model=list[FolderRunSummary])
def list_folder_runs(db: Session = Depends(get_db)) -> list[FolderRunSummary]:
    """Every folder run, newest first: the step-1 "Show recent runs" list.

    One row per source folder: duplicate runs pointing at the same folder
    (possible for runs created before the resume logic existed) collapse to
    the most recently updated one, mirroring ``_find_existing_run``. Without
    this, a list would surface duplicates that are invisible today.

    Not capped, and not paginated: the dedupe above needs the full set anyway,
    so a limit would save no work while hiding runs from the only surface that
    can open or delete them. The UI shows the most recent handful and reveals
    the rest on demand.

    ``folder_exists`` is stat'd per row so the UI can grey out runs whose
    folder moved or sits on an unplugged drive: those are not resumable, but
    can still be deleted.

    Declared above ``GET /{run_id}`` for the same route-ordering reason as
    ``/lookup`` (FastAPI matches in declaration order).
    """
    # A folder run has at most one queue entry, so this join yields one row
    # per run. Sorted newest-first, which makes the dedupe below "keep newest".
    rows = db.execute(
        select(Project, DeploymentQueue.folder_path, DeploymentQueue.status)
        .join(DeploymentQueue, DeploymentQueue.project_id == Project.id)
        .where(Project.mode == "folder_run")
        .order_by(Project.updated_at_utc.desc())
    ).all()

    picked: list[tuple[Project, str, str]] = []
    seen_folders: set[str] = set()
    for project, folder_path, queue_status in rows:
        if not folder_path or folder_path in seen_folders:
            continue
        seen_folders.add(folder_path)
        picked.append((project, folder_path, queue_status))

    if not picked:
        return []

    project_ids = [p.id for p, _, _ in picked]

    # One grouped count for the whole page rather than a query per row.
    file_counts = dict(
        db.execute(
            select(Deployment.project_id, func.count(File.id))
            .join(File, File.deployment_id == Deployment.id)
            .where(Deployment.project_id.in_(project_ids))
            .group_by(Deployment.project_id)
        ).all()
    )

    # Both halves of the "labels verified" fraction in one grouped query.
    # The denominator is the detections the run actually shows, so it carries
    # the threshold + verified override (DEVELOPERS.md "Detection threshold
    # and verified override"). The threshold is per-project, hence the join to
    # Project to compare against that run's own column rather than a scalar.
    # Verified detections always pass the filter, so they are a subset of the
    # counted set and can be summed in the same pass.
    label_counts = {
        project_id: (total, verified or 0)
        for project_id, total, verified in db.execute(
            select(
                Deployment.project_id,
                func.count(Detection.id),
                func.sum(case((Detection.verified.is_(True), 1), else_=0)),
            )
            .select_from(Detection)
            .join(File, Detection.file_id == File.id)
            .join(Deployment, File.deployment_id == Deployment.id)
            .join(Project, Deployment.project_id == Project.id)
            .where(Deployment.project_id.in_(project_ids))
            .where(threshold_or_verified(Project.counting_threshold))
            .group_by(Deployment.project_id)
        ).all()
    }

    summaries: list[FolderRunSummary] = []
    for project, folder_path, queue_status in picked:
        detection_count, verified_detection_count = label_counts.get(
            project.id, (0, 0)
        )
        summaries.append(
            FolderRunSummary(
                id=project.id,
                folder_path=folder_path,
                updated_at_utc=project.updated_at_utc,
                file_count=file_counts.get(project.id, 0),
                detection_count=detection_count,
                verified_detection_count=verified_detection_count,
                folder_exists=Path(folder_path).is_dir(),
                queue_status=queue_status,
            )
        )
    return summaries


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_folder_run(run_id: str, db: Session = Depends(get_db)) -> None:
    """Delete a folder run and everything it produced.

    Delegates to ``crud_project.delete_folder_run``, which cascades the DB
    rows AND removes the on-disk ``.addaxai/projects/<id>/`` cache.
    ``delete_project`` would leave that cache behind, so it must not be used
    here. Irreversible: any verification work in the run goes with it.

    404s for an unknown id or a research project, mirroring ``_load_run``.
    """
    project = crud_project.get_project(db, run_id)
    if project is None or project.mode != "folder_run":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder run with id '{run_id}' not found",
        )
    crud_project.delete_folder_run(db, run_id)
    logger.info(f"Deleted folder run: project_id={run_id}")


@router.get("/lookup", response_model=FolderRunLookupResponse | None)
def lookup_folder_run(
    folder: str,
    image_count: int | None = None,
    db: Session = Depends(get_db),
) -> FolderRunLookupResponse | None:
    """Probe for an existing folder-run project that points at ``folder``.

    Returns the summary the Step 1 picker needs to render the
    "already analysed" notice card. Returns ``null`` when no matching
    run exists — that's the common case, the form just proceeds.

    ``image_count`` is the picker's live scan of the folder. It decides
    whether an interrupted run's detection checkpoint still applies (see
    ``_detection_resume``); without it the queue entry's stored count is
    used, which is what the interrupted run saw.

    Declared above ``GET /{run_id}`` so the ``lookup`` literal is
    routed correctly (FastAPI matches routes in declaration order).
    """
    existing = _find_existing_run(db, folder)
    if existing is None:
        return None

    state: dict = existing.folder_run_state or {}
    step: FolderRunStep = _normalize_step(state.get("step", "setup"))

    # Cheap aggregates: total files, total threshold+verified
    # detections, distinct species label count, count of files with a
    # human verification on them. The threshold-or-verified rule is
    # the same one used everywhere else (DEVELOPERS.md "Detection
    # threshold and verified override").
    threshold = existing.counting_threshold
    deployment_ids_subq = (
        select(Deployment.id)
        .where(Deployment.project_id == existing.id)
        .scalar_subquery()
    )
    file_count = db.scalar(
        select(func.count(File.id)).where(
            File.deployment_id.in_(deployment_ids_subq)
        )
    ) or 0
    detection_filter = and_(
        File.deployment_id.in_(deployment_ids_subq),
        threshold_or_verified(threshold),
    )
    detection_count = db.scalar(
        select(func.count(Detection.id))
        .select_from(Detection)
        .join(File, Detection.file_id == File.id)
        .where(detection_filter)
    ) or 0
    species_count = db.scalar(
        select(func.count(distinct(Detection.label)))
        .select_from(Detection)
        .join(File, Detection.file_id == File.id)
        .where(detection_filter)
        .where(Detection.label.isnot(None))
    ) or 0
    verified_file_count = db.scalar(
        select(func.count(File.id))
        .where(File.deployment_id.in_(deployment_ids_subq))
        .where(File.verified.is_(True))
    ) or 0
    verified_detection_count = db.scalar(
        select(func.count(Detection.id))
        .select_from(Detection)
        .join(File, Detection.file_id == File.id)
        .where(detection_filter)
        .where(Detection.verified.is_(True))
    ) or 0
    # Count confirmation is the second half of the verification work
    # (events whose count was confirmed on the Counts page), mirroring the
    # dashboard's "Labels verified" + "Counts confirmed" split.
    event_count = db.scalar(
        select(func.count(Event.id)).where(
            Event.deployment_id.in_(deployment_ids_subq)
        )
    ) or 0
    confirmed_event_count = db.scalar(
        select(func.count(Event.id))
        .where(Event.deployment_id.in_(deployment_ids_subq))
        .where(Event.confirmed.is_(True))
    ) or 0

    detection_model_name = _friendly_model_name(existing.detection_model_id)
    classification_model_name = _friendly_model_name(
        existing.classification_model_id
    )

    return FolderRunLookupResponse(
        id=existing.id,
        name=existing.name,
        created_at_utc=existing.created_at_utc,
        updated_at_utc=existing.updated_at_utc,
        detection_model_id=existing.detection_model_id,
        classification_model_id=existing.classification_model_id,
        detection_model_name=detection_model_name,
        classification_model_name=classification_model_name,
        detection_image_size=existing.detection_image_size,
        detection_augment=existing.detection_augment,
        step=step,
        file_count=file_count,
        detection_count=detection_count,
        species_count=species_count,
        verified_file_count=verified_file_count,
        verified_detection_count=verified_detection_count,
        event_count=event_count,
        confirmed_event_count=confirmed_event_count,
        detection_resume=_detection_resume(db, existing, folder, image_count),
    )


def _detection_resume(
    db: Session, project: Project, folder: str, image_count: int | None
) -> DetectionResume | None:
    """What the interrupted run's checkpoint holds under the saved
    detection settings, or None.

    Only a ``failed`` entry is asked: a completed run has no checkpoint
    files left, and a pending one (after Cancel) continues on its own
    when it is started. The image count is the picker's live scan when
    it sent one, else the queue entry's, which the worker rewrote to
    what was on disk before detection started. A folder that gained or
    lost files since the crash therefore gets no Continue offer, which
    is also what the worker decides when it checks the same meta at run
    start.
    """
    if project.detection_model_id is None:
        return None
    entry = next(
        (
            e
            for e in crud_queue.get_queue_entries(db, project.id, status=None)
            if e.folder_path == folder
        ),
        None,
    )
    if entry is None or entry.status != "failed":
        return None
    state = ckpt.inspect(
        ckpt.artifacts_dir(Path(entry.folder_path), project.id),
        ckpt.CheckpointMeta(
            detection_model_id=project.detection_model_id,
            image_size=project.detection_image_size,
            augment=project.detection_augment,
            image_count=entry.image_count if image_count is None else image_count,
        ),
    )
    if state is None:
        return None
    return DetectionResume(
        images_done=state.images_done, images_total=state.images_total
    )


@router.get("/{run_id}", response_model=FolderRunResponse)
def get_folder_run(
    run_id: str, db: Session = Depends(get_db)
) -> FolderRunResponse:
    """Load an existing folder run by project id."""
    return _load_run(db, run_id)


@router.post("/{run_id}/rerun", response_model=FolderRunResponse)
def rerun_folder_run(
    run_id: str,
    body: RerunRequest | None = None,
    db: Session = Depends(get_db),
) -> FolderRunResponse:
    """Reset a folder run for re-analysis.

    Wipes the deployment / file / detection / event / embedding rows
    plus the on-disk ``.addaxai/projects/<id>/`` cache, and moves the
    queue entry back to ``status='pending'`` so the existing process
    endpoint picks it up. The project row and the queue entry id
    survive, so the URL stays valid and the persisted step stays put.
    With ``keep_checkpoint`` the interrupted run's detection checkpoint
    files stay in the cache so detection continues where it stopped.

    This destroys human verifications. The caller surfaces a
    destructive confirm dialog before invoking it.
    """
    project = crud_project.get_project(db, run_id)
    if project is None or project.mode != "folder_run":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder run with id '{run_id}' not found",
        )

    # Timed so a slow re-run can be attributed from the log alone. The
    # reset covers both the DB wipe and the on-disk .addaxai cleanup, and
    # the latter is unbounded on a slow external drive.
    started = time.perf_counter()
    ok = crud_project.reset_folder_run_data(
        db, run_id, keep_checkpoint=bool(body and body.keep_checkpoint)
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder run with id '{run_id}' not found",
        )

    logger.info(
        f"Reset folder run for re-analysis: project_id={run_id} "
        f"({time.perf_counter() - started:.1f}s)"
    )
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
        db,
        run_id,
        media_threshold=(
            payload.media_threshold if payload else DEFAULT_COUNTING_THRESHOLD
        ),
        excluded_label_ids=excluded,
        include_empty=bool(payload.include_empty) if payload else False,
        name_mode=payload.name_mode if payload else "common",
        group_events=payload.group_events if payload else True,
        group_by=payload.separate_group_by if payload else "flat",
        species_last=(
            payload.separate_species_last if payload else False
        ),
        videos_as_stills=bool(payload.anonymise) if payload else False,
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
            detail=f"Could not create output folder: {e}",
        ) from e
    # The scan-skip marker on addaxai-media is written by the WORKER,
    # never here. The worker wipes a marker-stamped media tree before
    # rebuilding it (retries must replace copies, not duplicate them),
    # and the marker is its proof of ownership. Stamping the folder
    # here would hand that proof to a pre-existing addaxai-media the
    # app never created, and the wipe would delete the user's files.
    job = job_crud.create_job(
        db,
        JobCreate(
            type="folder_run_save_outputs",
            # The whole request, not a hand-copied selection of it. This
            # used to transcribe all fourteen fields one by one, and a
            # transcription is a list somebody has to remember to extend:
            # `run_readme` was added to the schema's frontend twin and to
            # the worker, and never here, so the Save step's "Run details"
            # checkbox wrote the file whatever it was set to. Spreading
            # the model means a new flag reaches the worker by existing.
            #
            # `mode="json"` because `Job.payload` is a JSON column and the
            # spread copies whatever the model holds. Every field today is
            # a primitive, a list of strings or a Literal, so it changes
            # nothing; it is here so that the first field with a real type
            # (a datetime, a Path, an Enum) serialises instead of failing
            # at commit, far from the line that added it. Paying one word
            # for that is the point of spreading rather than transcribing.
            #
            # Both overrides are real differences rather than tidying:
            # run_id is not part of the request at all, and output_dir
            # goes through `Path` above, which is what created the folder
            # and what normalises a trailing slash.
            payload={
                **payload.model_dump(mode="json"),
                "run_id": run_id,
                "output_dir": str(output_root),
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
