"""Worker for the folder-run Save outputs job.

The Save step on the folder-run stepper kicks off a job rather than
running synchronously, so the frontend can render a blocking progress
modal with per-module status and a Cancel button. Each enabled module
runs sequentially in an executor; progress events are emitted on the
WebSocket between modules so the UI can update the checklist in real
time.

Cancellation is honoured between modules only — the individual modules
iterate full file lists internally and don't yet accept mid-loop
cancellation. That's a future refinement; the current between-module
check is good enough for the typical run.

Module sequencing matters: ``separate_folders`` runs first so the
shared ``OutputContext`` knows where each file ended up, then
``annotated_copies`` reads those placements and writes the
combined-effect image into each one. The data exports run last and
can pick up the resolved paths for the new ``relative_path`` column.

The job's ``result`` payload mirrors the per-module dataclass dicts so
the completion modal in the UI can render summary panels directly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.api.crud import job as job_crud
from app.api.crud import project as project_crud
from app.core.job_cancellation import (
    JobCancelledError,
    clear_cancel,
    is_cancel_requested,
)
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.postprocessing_outputs._output_context import OutputContext
from app.ml.postprocessing_outputs.annotated_copies import (
    write_annotated_copies,
)
from app.ml.postprocessing_outputs.observations_csv import (
    write_observations_csv,
)
from app.ml.postprocessing_outputs.observations_xlsx import (
    write_observations_xlsx,
)
from app.ml.postprocessing_outputs.recognition_json import (
    write_recognition_json,
)
from app.ml.postprocessing_outputs.run_readme import write_run_readme
from app.ml.postprocessing_outputs.separate_folders import (
    separate_into_folders,
)
from app.services.folder_scanner import OUTPUT_DIR_MARKER

logger = get_logger(__name__)


# Module ids — match the keys the frontend expects on the result payload
# and the labels in the progress events. The order here is the order
# the worker runs them in. ``separate_folders`` must come before
# ``annotated_copies`` so the shared ``OutputContext`` is populated
# before any module that consumes it.
_MODULE_ORDER: tuple[str, ...] = (
    "separate_folders",
    "annotated_copies",
    "recognition_json",
    "csv",
    "xlsx",
    "run_readme",
)

# User-facing names for the progress messages.
_MODULE_LABELS: dict[str, str] = {
    "separate_folders": "Separating files",
    "annotated_copies": "Writing annotated copies",
    "recognition_json": "Writing recognition JSON",
    "csv": "Writing CSV",
    "xlsx": "Writing XLSX",
    "run_readme": "Writing run README",
}


def _check_cancelled(job_id: str) -> None:
    if is_cancel_requested(job_id):
        raise JobCancelledError()


async def process_save_outputs_job(job_id: str) -> None:
    """Run the picked postprocess modules for a folder-run save job."""
    db = next(get_db())
    try:
        job = job_crud.get_job(db, job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        payload = job.payload or {}
        run_id = payload.get("run_id")
        output_dir = payload.get("output_dir")
        if not run_id or not output_dir:
            raise ValueError(
                "Missing run_id / output_dir on save-outputs job payload"
            )

        project = project_crud.get_project(db, run_id)
        if project is None or project.mode != "folder_run":
            raise ValueError(f"Folder run not found: {run_id}")

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        # Mark the output folder so future scans (preview + the analysis
        # worker's input enumeration) skip it — its separated / annotated
        # copies must never be re-ingested as input media. The save
        # endpoint also writes this, but doing it here too guarantees the
        # marker is co-located with the output tree this worker creates,
        # regardless of how the save was triggered. Best-effort.
        try:
            (output_root / OUTPUT_DIR_MARKER).touch(exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not write output marker in {output_root}: {e}")
        ctx = OutputContext(output_root=output_root)

        draw_bboxes = bool(payload.get("draw_bboxes"))
        anonymise = bool(payload.get("anonymise"))

        # Which modules actually fire on this run.
        active_modules: list[str] = []
        if payload.get("separate_folders"):
            active_modules.append("separate_folders")
        if draw_bboxes or anonymise:
            active_modules.append("annotated_copies")
        if payload.get("recognition_json"):
            active_modules.append("recognition_json")
        if payload.get("csv"):
            active_modules.append("csv")
        if payload.get("xlsx"):
            active_modules.append("xlsx")
        # README is always written.
        active_modules.append("run_readme")

        # Sort to match _MODULE_ORDER so the UI checklist sees a
        # consistent sequence regardless of dict-iteration quirks.
        active_modules.sort(key=lambda m: _MODULE_ORDER.index(m))
        total_modules = len(active_modules)

        # Resolve the per-call exclusion list once. The export
        # modules want a list of label name strings (not UUIDs); see
        # the folder_runs router for the rationale.
        from sqlalchemy import func, select

        from app.models import Deployment, File

        # The label filter scopes the MEDIA outputs only (separate /
        # visualise / anonymise). Data exports (CSV / XLSX / recognition
        # JSON) are the complete record of the run and always include
        # every label.
        raw_excluded = payload.get("excluded_label_ids") or []
        excluded_frozen = (
            frozenset(raw_excluded) if raw_excluded else None
        )

        job_crud.update_job_status(db, job_id, "running")
        await _emit_module_event(
            job_id,
            module=None,
            module_index=0,
            total_modules=total_modules,
            active_modules=active_modules,
            status="Starting",
        )

        loop = asyncio.get_event_loop()
        # Count source files once up front so the completion screen can
        # show a single "N source files processed" tally without
        # double-counting multi-placement inflation in any per-module
        # number.
        source_file_count = db.scalar(
            select(func.count(File.id))
            .join(Deployment, File.deployment_id == Deployment.id)
            .where(Deployment.project_id == project.id)
        ) or 0

        result_payload: dict[str, Any] = {
            "output_dir": str(output_root),
            "source_file_count": int(source_file_count),
        }

        for idx, module in enumerate(active_modules):
            _check_cancelled(job_id)
            label = _MODULE_LABELS.get(module, module)
            await _emit_module_event(
                job_id,
                module=module,
                module_index=idx,
                total_modules=total_modules,
                active_modules=active_modules,
                status=label,
            )

            def _run(m: str = module) -> dict[str, Any]:
                if m == "separate_folders":
                    return separate_into_folders(
                        db,
                        project.id,
                        ctx,
                        mode="copy",
                        group_by=payload.get(
                            "separate_group_by", "flat"
                        ),
                        include_empty=bool(
                            payload.get("include_empty", False)
                        ),
                        excluded_label_ids=excluded_frozen,
                        name_mode=payload.get("name_mode", "common"),
                    ).to_dict()
                if m == "annotated_copies":
                    return write_annotated_copies(
                        db,
                        project.id,
                        ctx,
                        draw_bboxes=draw_bboxes,
                        anonymise=anonymise,
                        excluded_label_ids=excluded_frozen,
                        name_mode=payload.get("name_mode", "common"),
                    ).to_dict()
                if m == "recognition_json":
                    return write_recognition_json(
                        db,
                        project.id,
                        ctx.output_root,
                    ).to_dict()
                if m == "csv":
                    return write_observations_csv(
                        db,
                        project.id,
                        ctx,
                    ).to_dict()
                if m == "xlsx":
                    return write_observations_xlsx(
                        db,
                        project.id,
                        ctx,
                    ).to_dict()
                if m == "run_readme":
                    return write_run_readme(
                        db, project.id, ctx.output_root
                    ).to_dict()
                raise ValueError(f"Unknown module: {m}")

            module_result = await loop.run_in_executor(None, _run)
            result_payload[module] = module_result

        # Persist the full result on the job so the UI can pull it
        # later if the WebSocket dropped before the complete event.
        refreshed = job_crud.get_job(db, job_id)
        if refreshed is not None:
            refreshed.result = result_payload
            db.commit()

        await _emit_module_event(
            job_id,
            module=None,
            module_index=total_modules,
            total_modules=total_modules,
            active_modules=active_modules,
            status="Done",
        )
        job_crud.update_job_status(db, job_id, "completed")
        await ws_manager.send_complete(
            job_id=job_id,
            success=True,
            message="Save outputs complete",
            data=result_payload,
        )
        logger.info(
            f"folder_run_save_outputs job {job_id} done: "
            f"modules={active_modules}"
        )

    except JobCancelledError:
        logger.info(f"folder_run_save_outputs job {job_id} cancelled")
        job_crud.update_job_status(db, job_id, "cancelled")
        await ws_manager.send_cancelled(
            job_id, "Save cancelled by user"
        )
    except Exception as e:
        logger.exception(
            f"folder_run_save_outputs job {job_id} failed: {e}"
        )
        try:
            job_crud.update_job_status(db, job_id, "failed")
        except Exception:
            pass
        await ws_manager.send_error(job_id, str(e))
    finally:
        clear_cancel(job_id)
        db.close()


async def _emit_module_event(
    job_id: str,
    *,
    module: str | None,
    module_index: int,
    total_modules: int,
    active_modules: list[str],
    status: str,
) -> None:
    """Send a progress message that the frontend's JobProgressModal
    turns into per-module checklist state.

    The ``data`` dict carries:
    - ``current_module``: id of the module currently running, or None
      at the start / end of the job
    - ``modules``: ordered list of module ids so the UI can render
      the full checklist
    - ``module_index``: how many modules have completed so far
    """
    progress = (
        module_index / total_modules if total_modules > 0 else 0.0
    )
    await ws_manager.send_progress(
        job_id,
        f"{status} ({module_index} / {total_modules})",
        progress,
        data={
            "current_module": module,
            "module_index": module_index,
            "total_modules": total_modules,
            "modules": active_modules,
        },
    )
