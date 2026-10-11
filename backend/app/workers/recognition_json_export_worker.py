"""Recognition-file export worker: one addaxai-recognitions.json per
deployment folder, with progress.

A project can hold hundreds of deployments on several drives, so this
runs as a job with per-deployment progress over the existing ws_manager
/ useTaskProgress plumbing, like the Camtrap DP export. A deployment
whose folder is missing or cannot be written is reported in the result,
not raised: the other folders are still worth writing. The result (what
was written, what was skipped and why) is the completion payload and is
kept on the job row.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.api.crud import job as job_crud
from app.api.schemas.export import (
    RecognitionJsonExportResult,
    RecognitionJsonSkipped,
    RecognitionJsonWritten,
)
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.postprocessing_outputs.recognition_json import write_recognition_json
from app.models import Deployment

logger = get_logger(__name__)


def _write_one(
    db, project_id: str, dep: Deployment
) -> RecognitionJsonWritten | RecognitionJsonSkipped:
    folder = Path(dep.folder_path) if dep.folder_path else None
    if folder is None or not folder.is_dir():
        return RecognitionJsonSkipped(
            deployment_id=dep.id,
            folder_path=dep.folder_path,
            reason="Folder not found" if folder else "No folder linked",
        )
    try:
        written = write_recognition_json(
            db, project_id, folder, deployment_id=dep.id
        )
    except OSError as e:
        logger.warning(f"recognition-json export: could not write into {folder}: {e}")
        return RecognitionJsonSkipped(
            deployment_id=dep.id,
            folder_path=dep.folder_path,
            reason=f"Could not write: {e.strerror or e}",
        )
    return RecognitionJsonWritten(
        deployment_id=dep.id,
        folder_path=dep.folder_path,
        output_path=written.output_path,
        image_count=written.image_count,
    )


async def process_recognition_json_export_job(job_id: str) -> None:
    db = next(get_db())
    try:
        job = job_crud.get_job(db, job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        payload = job.payload or {}
        project_id = payload.get("project_id")
        if not project_id:
            raise ValueError("Missing project_id in job payload")
        # None means the whole project; a list is the resolved scope.
        deployment_ids = payload.get("deployment_ids")

        job_crud.update_job_status(db, job_id, "running")

        query = db.query(Deployment).filter(Deployment.project_id == project_id)
        if deployment_ids is not None:
            query = query.filter(Deployment.id.in_(deployment_ids))
        deployments = query.order_by(Deployment.folder_path).all()
        total = len(deployments)

        result = RecognitionJsonExportResult(written=[], skipped=[])
        loop = asyncio.get_event_loop()
        for i, dep in enumerate(deployments, start=1):
            await ws_manager.send_progress(
                job_id,
                f"Writing {Path(dep.folder_path).name if dep.folder_path else '?'}"
                f" ({i}/{total})",
                (i - 1) / total,
                data={"metrics": {"current": i, "total": total, "unit": "deployment"}},
            )
            # The write is minutes of SQL and JSON on a big deployment;
            # off the loop so the progress above actually reaches the UI.
            outcome = await loop.run_in_executor(None, _write_one, db, project_id, dep)
            if isinstance(outcome, RecognitionJsonWritten):
                result.written.append(outcome)
            else:
                result.skipped.append(outcome)

        refreshed = job_crud.get_job(db, job_id)
        if refreshed is not None:
            refreshed.payload = {**(refreshed.payload or {}), "result": result.model_dump()}
            db.commit()
        job_crud.update_job_status(db, job_id, "completed")
        await ws_manager.send_complete(
            job_id=job_id,
            success=True,
            message="Recognition files written",
            data=result.model_dump(),
        )
        logger.info(
            f"recognition-json export job {job_id}: {len(result.written)} written, "
            f"{len(result.skipped)} skipped"
        )
    except Exception as e:
        logger.error(f"recognition-json export job {job_id} failed: {e}", exc_info=True)
        try:
            job_crud.update_job_status(db, job_id, "failed")
        except Exception:
            pass
        await ws_manager.send_error(job_id, str(e))
    finally:
        db.close()
