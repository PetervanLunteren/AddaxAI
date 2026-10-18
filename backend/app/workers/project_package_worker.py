"""Project package jobs: export a package to a folder, import one.

Both run the blocking work (SQLite copies, a media tree, alembic on a
copy) in an executor thread through one runner, which relays progress
over the job WebSocket and stores the result, the shape of the other
export workers. Export can be cancelled between tables and between
copied files, and cleans up after itself. Import is one transaction and
runs to the end once started.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.crud import job as job_crud
from app.api.schemas.job import JobUpdate
from app.core.config import Settings, get_settings
from app.core.job_cancellation import (
    JobCancelledError,
    clear_cancel,
    is_cancel_requested,
)
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.backup import live_db_path
from app.db.base import get_db
from app.services.project_package import (
    ProgressFn,
    export_package,
    import_package,
)

logger = get_logger(__name__)

# `work(db, settings, payload, report)`, run in the executor thread.
Work = Callable[[Session, Settings, dict[str, Any], ProgressFn], BaseModel]


def _progress_relay(
    job_id: str, loop: asyncio.AbstractEventLoop, *, cancellable: bool
) -> ProgressFn:
    """A `report(message, done, total)` for the service, called from the
    executor thread. Throttled to four sends a second, the final tick
    always sent. Raises `JobCancelledError` on a cancel request when the
    job can stop."""
    state = {"t": 0.0}

    def report(message: str, done: int, total: int) -> None:
        if cancellable and is_cancel_requested(job_id):
            raise JobCancelledError()
        now = time.monotonic()
        if done < total and now - state["t"] < 0.25:
            return
        state["t"] = now
        asyncio.run_coroutine_threadsafe(
            ws_manager.send_progress(
                job_id,
                message,
                done / total if total else 0.0,
                data={"metrics": {"current": done, "total": total}},
            ),
            loop,
        )

    return report


async def _run_job(
    job_id: str, work: Work, *, cancellable: bool, done_message: str
) -> None:
    """Run one package job from "ready" to a stored result."""
    db = next(get_db())
    try:
        job = job_crud.get_job(db, job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        payload = job.payload or {}
        job_crud.update_job_status(db, job_id, "running")
        loop = asyncio.get_event_loop()
        report = _progress_relay(job_id, loop, cancellable=cancellable)
        settings = get_settings()

        result = await loop.run_in_executor(
            None, lambda: work(db, settings, payload, report)
        )
        data = result.model_dump()
        job_crud.update_job(db, job_id, JobUpdate(status="completed", result=data))
        await ws_manager.send_complete(
            job_id, success=True, message=done_message, data=data
        )
    except JobCancelledError:
        logger.info(f"Package job {job_id} cancelled")
        job_crud.update_job_status(db, job_id, "cancelled")
        await ws_manager.send_cancelled(job_id, "Export cancelled")
    except Exception as e:
        logger.error(f"Package job {job_id} failed: {e}", exc_info=True)
        db.rollback()
        job_crud.update_job(db, job_id, JobUpdate(status="failed", error=str(e)))
        await ws_manager.send_error(job_id, str(e))
    finally:
        clear_cancel(job_id)
        db.close()


def _export(
    db: Session, settings: Settings, payload: dict[str, Any], report: ProgressFn
) -> BaseModel:
    return export_package(
        live_db_path(settings),
        payload["project_id"],
        payload["deployment_ids"],
        Path(payload["output_dir"]),
        include_media=bool(payload.get("include_media")),
        report=report,
    )


def _import(
    db: Session, settings: Settings, payload: dict[str, Any], report: ProgressFn
) -> BaseModel:
    result = import_package(
        db,
        settings,
        Path(payload["path"]),
        replace_existing=bool(payload.get("replace_existing")),
        report=report,
    )
    _pick_thumbnail(db, settings, result.project_id)
    return result


def _pick_thumbnail(db: Session, settings: Settings, project_id: str) -> None:
    """Give an imported project its card picture now rather than at the
    next launch. Only that project: the startup pass re-rolls every
    automatic thumbnail, which an import has no business doing. Nice to
    have: a failure is logged and the import stands."""
    try:
        from app.models import Project
        from app.services.thumbnail_service import _auto_select_for_project

        project = db.get(Project, project_id)
        if project is None or project.thumbnail_path:
            return
        thumbnails_dir = settings.user_data_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        _auto_select_for_project(db, project, thumbnails_dir)
    except Exception as e:
        db.rollback()
        logger.warning(f"Could not pick a thumbnail after import: {e}")


async def process_package_export_job(job_id: str) -> None:
    await _run_job(job_id, _export, cancellable=True, done_message="Bundle written")


async def process_package_import_job(job_id: str) -> None:
    await _run_job(job_id, _import, cancellable=False, done_message="Bundle imported")
