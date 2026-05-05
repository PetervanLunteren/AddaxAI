"""
Timelapse Analyser integration router.

Single endpoint that kicks off a DB-less analysis run on a folder. Progress
streams over the existing `/ws/jobs/{job_id}` websocket via `ws_manager`,
so the frontend's `useTaskProgress` hook works unchanged.

The handler returns immediately with a `job_id`. The actual run executes
on a background asyncio task. No Job row is created — Timelapse runs are
intentionally invisible to the user's main-app project DB.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.ml.timelapse_runner import TimelapseRunRequest
from app.ml.timelapse_runner import run as run_timelapse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/timelapse", tags=["timelapse"])


class TimelapseRunBody(BaseModel):
    folder_path: str
    classification_model_id: str | None = None
    detection_model_id: str = "MD5A-0-0"
    excluded_classes: list[str] = Field(default_factory=list)
    # Detection confidence is intentionally NOT exposed in Timelapse mode.
    # The runner uses the shared DETECTION_CONFIDENCE_FLOOR (in
    # app/ml/detection.py), matching what the main app's worker passes,
    # so users can do their own filtering inside Timelapse Analyser.
    #
    # None means "use the subprocess's built-in default", same convention
    # as Project.detection_batch_size / classification_batch_size.
    detection_batch_size: int | None = None
    classification_batch_size: int | None = None
    video_fps: float = 1.0
    # independence_interval is intentionally NOT exposed. It only feeds
    # the sequence-level smoother; the Timelapse runner uses
    # TIMELAPSE_INDEPENDENCE_INTERVAL (1800s, matching the main app default).
    smoothing_strength: Literal["off", "mild", "normal", "aggressive"] = "normal"
    taxonomic_rollup: bool = True


class TimelapseRunResponse(BaseModel):
    job_id: str


async def _run_and_handle_errors(req: TimelapseRunRequest, job_id: str) -> None:
    """Wrap the runner so any exception lands as a websocket error.

    The runner itself emits `send_complete(success=True, ...)` on the happy
    path. If it raises, we forward the message verbatim so the frontend's
    error UI surfaces something actionable instead of a hung progress bar.
    """
    try:
        await run_timelapse(req, job_id)
    except Exception as e:
        logger.error(f"Timelapse run {job_id} failed: {e}", exc_info=True)
        try:
            await ws_manager.send_complete(
                job_id, False, f"Timelapse run failed: {e}", data={"error": str(e)}
            )
        except Exception:
            pass


@router.post("/run", response_model=TimelapseRunResponse)
async def start_timelapse_run(body: TimelapseRunBody) -> TimelapseRunResponse:
    """Start a Timelapse analysis run on the given folder.

    Returns a job_id the client uses to subscribe to `/ws/jobs/{job_id}`
    for progress and the final output path.
    """
    folder = Path(body.folder_path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Folder does not exist: {body.folder_path}",
        )

    job_id = str(uuid.uuid4())
    req = TimelapseRunRequest(
        folder_path=folder,
        classification_model_id=body.classification_model_id,
        detection_model_id=body.detection_model_id,
        excluded_classes=body.excluded_classes or None,
        detection_batch_size=body.detection_batch_size,
        classification_batch_size=body.classification_batch_size,
        video_fps=body.video_fps,
        smoothing_strength=body.smoothing_strength,
        taxonomic_rollup=body.taxonomic_rollup,
    )

    asyncio.create_task(_run_and_handle_errors(req, job_id))
    logger.info(f"Started Timelapse run {job_id} on {folder}")
    return TimelapseRunResponse(job_id=job_id)
