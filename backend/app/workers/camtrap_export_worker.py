"""Camtrap DP export worker: builds a Camtrap DP ZIP with progress and
writes it into the folder the user picked, as
``camtrap-dp-<project>-<date>.zip`` (``_2`` and up when the name is taken).

Runs async so the frontend can show real-time progress via the existing
ws_manager / useTaskProgress plumbing. The package itself is built in a
worker thread, row by row, into a hidden staging folder inside the output
folder and then zipped from disk: Camtrap DP keeps every video frame, so it is the largest table
export, and nothing of it is ever held in memory (``CamtrapDpBuilder``).
"""

from __future__ import annotations

import asyncio
import json as _json
import tempfile
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.crud import export as export_crud
from app.api.crud import export_formats
from app.api.crud import job as job_crud
from app.api.schemas.job import JobUpdate
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.models import Project
from app.utils.datetime_serialization import set_active_project_timezone

logger = get_logger(__name__)

# Throttle progress emissions so the client's 16 ms debounce actually
# flushes between updates (without throttling, rapid-fire per-file sends
# keep resetting the debounce and the UI never repaints until the job
# completes).
_MIN_EMIT_INTERVAL = 0.1  # seconds


def _thumbnail_progress(
    job_id: str, loop: asyncio.AbstractEventLoop
) -> Callable[[int, int], None]:
    """Per-file progress relay for the thumbnail pass, which runs in a
    worker thread: throttled, and sent on the event loop."""
    last_emit = 0.0

    def report(done: int, total: int) -> None:
        nonlocal last_emit
        now = loop.time()
        if done != total and now - last_emit < _MIN_EMIT_INTERVAL:
            return
        last_emit = now
        frac = min(1.0, done / total) if total else 1.0
        asyncio.run_coroutine_threadsafe(
            ws_manager.send_progress(
                job_id,
                f"Generating thumbnails ({done}/{total})",
                frac,
                phase="thumbnails",
                phase_progress=frac,
                data={"metrics": {"current": done, "total": total, "unit": "file"}},
            ),
            loop,
        )

    return report


def _stage_package(
    db: Session,
    project: Project,
    deployment_ids: list[str] | None,
    staging: Path,
    on_thumbnail: Callable[[int, int], None] | None,
) -> tuple[int, int]:
    """Write the four package files (and thumbnails) into ``staging``.
    Returns ``(media rows, thumbnails)``."""
    builder = export_crud.CamtrapDpBuilder(db, project, deployment_ids)
    deps_h, media_h, obs_h = export_crud.camtrap_dp_headers()
    export_formats.write_csv(staging / "deployments.csv", deps_h, builder.deployments_rows)

    total = builder.media_count() if on_thumbnail else 0
    media_dir = staging / "media"
    n_media = n_thumbnails = 0
    with (
        export_formats.csv_writer(staging / "media.csv", media_h) as media_out,
        export_formats.csv_writer(staging / "observations.csv", obs_h) as obs_out,
    ):
        for media_row, observation_rows in builder.media():
            n_media += 1
            if on_thumbnail is not None:
                media_id, source_path = media_row[0], media_row[4]
                thumb = export_formats.generate_thumbnail(source_path)
                if thumb is not None:
                    media_dir.mkdir(exist_ok=True)
                    (media_dir / f"{media_id}.jpg").write_bytes(thumb)
                    n_thumbnails += 1
                    # Keep index references in sync with _CAMTRAP_MEDIA_HEADERS.
                    media_row[4] = f"media/{media_id}.jpg"
                    media_row[7] = "image/jpeg"
                on_thumbnail(n_media, total)
            media_out.writerow(media_row)
            obs_out.writerows(observation_rows)
        obs_out.writerows(builder.event_observations())

    (staging / "datapackage.json").write_bytes(
        _json.dumps(builder.datapackage(), indent=2, ensure_ascii=False).encode("utf-8")
    )
    return n_media, n_thumbnails


async def process_camtrap_export_job(job_id: str) -> None:
    db = next(get_db())
    try:
        job = job_crud.get_job(db, job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        payload = job.payload or {}
        project_id = payload.get("project_id")
        include_thumbnails = bool(payload.get("include_thumbnails", False))
        deployment_ids = payload.get("deployment_ids")
        output_dir = payload.get("output_dir")
        if not project_id or not output_dir:
            raise ValueError("Missing project_id / output_dir in job payload")

        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            raise ValueError(f"Project not found: {project_id}")
        if project.timezone:
            set_active_project_timezone(project.timezone)

        job_crud.update_job_status(db, job_id, "running")
        await ws_manager.send_progress(job_id, "Building tables...", 0.0)

        loop = asyncio.get_running_loop()
        on_thumbnail = _thumbnail_progress(job_id, loop) if include_thumbnails else None
        # Staged inside the output folder, on the drive the user picked and
        # checked for space, rather than in the system temp folder: a nearly
        # full system drive failed the export with "No space left on
        # device" while the chosen folder had room (2026-10-15). The
        # staged files are about the uncompressed package, and the hidden
        # folder goes again when the ZIP is written, or on failure.
        with tempfile.TemporaryDirectory(
            dir=output_dir, prefix=".addaxai-camtrap-"
        ) as tmp:
            staging = Path(tmp)
            # Off the event loop: a large project must not freeze every
            # other request. The thread inherits the project timezone.
            n_media, n_thumbnails = await asyncio.to_thread(
                _stage_package, db, project, deployment_ids, staging, on_thumbnail
            )

            await ws_manager.send_progress(job_id, "Packaging ZIP...", 0.99)
            path = export_formats.unique_export_path(
                Path(output_dir),
                export_formats.export_file_name(project.name, "camtrap-dp", "zip"),
            )
            # Written under a .partial name and renamed when complete, so
            # the user's folder never shows half a ZIP under the real name,
            # not even while a large package is being written.
            partial = path.with_name(path.name + ".partial")
            try:
                await asyncio.to_thread(export_formats.write_camtrap_dp_zip, partial, staging)
                partial.replace(path)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        result = {"output_dir": output_dir, "files": [path.name]}

        job_crud.update_job(
            db, job_id, JobUpdate(status="completed", result=result)
        )
        await ws_manager.send_complete(
            job_id=job_id, success=True, message="Export written", data=result
        )
        logger.info(
            f"CamtrapDP export job {job_id} done: {n_media} media rows, "
            f"thumbnails={n_thumbnails}, path={path}"
        )

    except Exception as e:
        logger.error(f"CamtrapDP export job {job_id} failed: {e}", exc_info=True)
        try:
            job_crud.update_job_status(db, job_id, "failed")
        except Exception:
            pass
        await ws_manager.send_error(job_id, str(e))
    finally:
        db.close()
