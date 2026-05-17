"""Worker for the folder-run Save outputs job.

The Save step on the folder-run stepper kicks off a job rather than
running synchronously, so the frontend can render a blocking progress
modal with per-module status and a Cancel button. Each enabled
module (Separate / Visualise / Blur / EXIF / CSV / XLSX / Recognition
JSON / README) runs sequentially in an executor; progress events are
emitted on the WebSocket between modules so the UI can update the
checklist in real time.

Cancellation is honoured between modules only — the individual
modules iterate full file lists internally and don't yet accept
mid-loop cancellation. That's a future refinement; the current
between-module check is good enough for the typical run (one slow
module of a few minutes, not hours).

The job's ``result`` payload mirrors the shape the synchronous
endpoint used to return so the completion modal in the UI can
render the same per-module summary panels with no shape change.
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
from app.ml.postprocessing_outputs.blur_people import blur_people
from app.ml.postprocessing_outputs.exif_metadata import (
    write_exif_predictions,
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
from app.ml.postprocessing_outputs.visualised_images import (
    visualise_images,
)

logger = get_logger(__name__)


# Module ids — match the keys the frontend expects on the result
# payload + the labels in the progress events. The order here is
# the order the worker runs them in.
_MODULE_ORDER: tuple[str, ...] = (
    "separate_folders",
    "visualised_images",
    "blur_people",
    "write_exif",
    "recognition_json",
    "csv",
    "xlsx",
    "run_readme",
)

# User-facing names for the progress messages.
_MODULE_LABELS: dict[str, str] = {
    "separate_folders": "Separating files",
    "visualised_images": "Visualising detections",
    "blur_people": "Blurring people and vehicles",
    "write_exif": "Writing EXIF tags",
    "recognition_json": "Writing recognition JSON",
    "csv": "Writing CSV",
    "xlsx": "Writing XLSX",
    "run_readme": "Writing run README",
}


def _check_cancelled(job_id: str) -> None:
    if is_cancel_requested(job_id):
        raise JobCancelledError()


async def process_save_outputs_job(job_id: str) -> None:
    """Run the picked postprocess modules for a folder-run save job.

    Each module runs synchronously in an executor so the asyncio
    loop stays free to flush WebSocket progress events between
    them. The job payload carries every flag and parameter the
    synchronous endpoint used to take; the result payload mirrors
    the synchronous response shape.
    """
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

        # Resolve which modules to run. Same flags as the
        # synchronous endpoint accepted.
        active_modules: list[str] = []
        if payload.get("separate_folders"):
            active_modules.append("separate_folders")
        if payload.get("visualised_images"):
            active_modules.append("visualised_images")
        if payload.get("blur_people"):
            active_modules.append("blur_people")
        if payload.get("write_exif"):
            active_modules.append("write_exif")
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
        # modules want a list of label name strings (not UUIDs);
        # see folder_runs router for the rationale.
        from app.models import LabelTaxonomy
        from sqlalchemy import select

        raw_excluded = payload.get("excluded_label_ids") or []
        excluded_frozen = (
            frozenset(raw_excluded) if raw_excluded else None
        )
        excluded_names_for_exports: list[str] = []
        if raw_excluded:
            resolved = db.execute(
                select(LabelTaxonomy.id, LabelTaxonomy.name).where(
                    LabelTaxonomy.id.in_(raw_excluded)
                )
            ).all()
            resolved_ids = {r.id for r in resolved}
            excluded_names_for_exports = [r.name for r in resolved]
            excluded_names_for_exports.extend(
                item for item in raw_excluded if item not in resolved_ids
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
        result_payload: dict[str, Any] = {
            "output_dir": str(output_root),
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
                    target = output_root / "separated"
                    return separate_into_folders(
                        db,
                        project.id,
                        target,
                        mode=payload.get("separate_method", "copy"),
                        group_by=payload.get(
                            "separate_group_by", "taxonomic"
                        ),
                        excluded_label_ids=excluded_frozen,
                    ).to_dict()
                if m == "visualised_images":
                    target = output_root / "visualised"
                    return visualise_images(
                        db,
                        project.id,
                        target,
                        excluded_label_ids=excluded_frozen,
                    ).to_dict()
                if m == "blur_people":
                    target = output_root / "blurred"
                    return blur_people(
                        db,
                        project.id,
                        target,
                        excluded_label_ids=excluded_frozen,
                    ).to_dict()
                if m == "write_exif":
                    return write_exif_predictions(
                        db,
                        project.id,
                        output_root,
                        mode=payload.get("exif_mode", "copy"),
                        excluded_label_ids=excluded_frozen,
                    ).to_dict()
                if m == "recognition_json":
                    return write_recognition_json(
                        db,
                        project.id,
                        output_root,
                        excluded_species=excluded_names_for_exports,
                    ).to_dict()
                if m == "csv":
                    return write_observations_csv(
                        db,
                        project.id,
                        output_root,
                        excluded_species=excluded_names_for_exports,
                    ).to_dict()
                if m == "xlsx":
                    return write_observations_xlsx(
                        db,
                        project.id,
                        output_root,
                        excluded_species=excluded_names_for_exports,
                    ).to_dict()
                if m == "run_readme":
                    return write_run_readme(
                        db, project.id, output_root
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
