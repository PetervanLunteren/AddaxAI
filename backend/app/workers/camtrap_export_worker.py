"""Camtrap DP export worker: builds a Camtrap DP ZIP with progress and
writes it into the folder the user picked, as
``camtrap-dp-<project>-<date>.zip`` (``_2`` and up when the name is taken).

Runs async so the frontend can show real-time progress via the existing
ws_manager / useTaskProgress plumbing.
"""

from __future__ import annotations

import asyncio
import json as _json
from pathlib import Path

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

        scoped = export_crud.get_scoped_detection_rows(
            db, project, deployment_ids=deployment_ids
        )
        (
            deps_rows,
            media_rows,
            obs_rows,
            datapackage,
            skipped_deployment_ids,
        ) = export_crud.build_camtrap_dp_tables(db, project, scoped, deployment_ids)
        deps_h, media_h, obs_h = export_crud.camtrap_dp_headers()

        thumbnails: dict[str, bytes] | None = None
        if include_thumbnails:
            thumbnails = {}
            total = len(media_rows)
            loop = asyncio.get_event_loop()

            # Seed a 0/total progress event before the first thumbnail
            # lands so the UI shows the scale immediately instead of
            # "Starting...".
            await ws_manager.send_progress(
                job_id,
                f"Generating thumbnails (0/{total})",
                0.0,
                phase="thumbnails",
                phase_progress=0.0,
                data={"metrics": {"current": 0, "total": total, "unit": "file"}},
            )

            # Throttle progress emissions so the client's 16 ms debounce
            # actually flushes between updates (without throttling,
            # rapid-fire per-file sends keep resetting the debounce and
            # the UI never repaints until the job completes).
            last_emit_monotonic = loop.time()
            _MIN_EMIT_INTERVAL = 0.1  # 100 ms

            for i, row in enumerate(media_rows, start=1):
                media_id = row[0]
                source_path = row[4]
                # Run Pillow on a thread so the event loop stays free
                # to flush WebSocket progress events in real time.
                thumb_bytes = await loop.run_in_executor(
                    None, export_formats.generate_thumbnail, source_path
                )
                if thumb_bytes is not None:
                    rel_name = f"{media_id}.jpg"
                    thumbnails[rel_name] = thumb_bytes
                    # Keep index references in sync with _CAMTRAP_MEDIA_HEADERS.
                    row[4] = f"media/{rel_name}"
                    row[7] = "image/jpeg"

                now = loop.time()
                if (now - last_emit_monotonic) >= _MIN_EMIT_INTERVAL or i == total:
                    last_emit_monotonic = now
                    frac = i / total
                    await ws_manager.send_progress(
                        job_id,
                        f"Generating thumbnails ({i}/{total})",
                        frac,
                        phase="thumbnails",
                        phase_progress=frac,
                        data={
                            "metrics": {
                                "current": i,
                                "total": total,
                                "unit": "file",
                            }
                        },
                    )

        await ws_manager.send_progress(job_id, "Packaging ZIP...", 0.99)

        deps_csv = export_formats.serialize_csv(deps_h, deps_rows)
        media_csv = export_formats.serialize_csv(media_h, media_rows)
        obs_csv = export_formats.serialize_csv(obs_h, obs_rows)
        datapackage_bytes = _json.dumps(
            datapackage, indent=2, ensure_ascii=False
        ).encode("utf-8")
        zip_bytes = export_formats.build_camtrap_dp_zip(
            datapackage_bytes,
            deps_csv,
            media_csv,
            obs_csv,
            thumbnails=thumbnails,
        )

        path = export_formats.unique_export_path(
            Path(output_dir),
            export_formats.export_file_name(project.name, "camtrap-dp", "zip"),
        )
        path.write_bytes(zip_bytes)
        result = {"output_dir": output_dir, "files": [path.name]}

        job_crud.update_job(
            db, job_id, JobUpdate(status="completed", result=result)
        )
        await ws_manager.send_complete(
            job_id=job_id, success=True, message="Export written", data=result
        )
        logger.info(
            f"CamtrapDP export job {job_id} done: {len(media_rows)} media rows, "
            f"thumbnails={len(thumbnails) if thumbnails else 0}, path={path}"
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
