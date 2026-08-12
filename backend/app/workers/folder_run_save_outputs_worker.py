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
combined-effect image into each one. Media copies land under the
``addaxai-media`` subfolder of the output dir; the loose data exports
(``addaxai-*.csv`` / ``.xlsx`` / ``.json`` / ``.txt``) go at the
output root, which defaults to the source folder itself.

The job's ``result`` payload mirrors the per-module dataclass dicts so
the completion modal in the UI can render summary panels directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.api.crud import job as job_crud
from app.api.crud import project as project_crud
from app.core.confidence import DEFAULT_COUNTING_THRESHOLD
from app.core.job_cancellation import (
    JobCancelledError,
    clear_cancel,
    is_cancel_requested,
)
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.postprocessing_outputs._output_context import (
    MEDIA_SUBDIR,
    OutputContext,
)
from app.ml.postprocessing_outputs.annotated_copies import (
    write_annotated_copies,
)
from app.ml.postprocessing_outputs.recognition_json import (
    write_recognition_json,
)
from app.ml.postprocessing_outputs.run_readme import write_run_readme
from app.ml.postprocessing_outputs.separate_folders import (
    separate_into_folders,
)
from app.ml.postprocessing_outputs.tables_csv import (
    write_tables_csv,
)
from app.ml.postprocessing_outputs.tables_xlsx import (
    write_tables_xlsx,
)
from app.services.folder_scanner import OUTPUT_DIR_MARKER
from app.utils.fs_remove import safe_rmtree
from app.utils.process_memory import rss_mb

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
    "run_readme": "Writing run details",
}


def _rss_for_log() -> str:
    """Current RSS formatted for a log line; 'unknown' when unreadable."""
    value = rss_mb()
    return f"{value:.0f}" if value is not None else "unknown"


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

        # Layout: loose ``addaxai-*`` data files at the output root
        # (which defaults to the source folder itself), media copies
        # under the ``addaxai-media`` subfolder.
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        media_root = output_root / MEDIA_SUBDIR
        ctx = OutputContext(output_root=media_root)

        draw_bboxes = bool(payload.get("draw_bboxes"))
        anonymise = bool(payload.get("anonymise"))
        # Media-output confidence from the Save step. Scopes the media
        # modules only; the data exports are always the complete record.
        # Falls back to the schema default for jobs enqueued by an older
        # frontend that did not send the field.
        media_threshold = float(
            payload.get("media_threshold", DEFAULT_COUNTING_THRESHOLD)
        )
        media_active = bool(
            payload.get("separate_folders") or draw_bboxes or anonymise
        )
        if media_active:
            # Rebuild the media tree from scratch. A save always places
            # copies (mode="copy" below), so the tree holds nothing the
            # user can lose, and without this every retry of a failed
            # save re-copies each file under a `_2` / `_3` name (the
            # destinations from the earlier attempt already exist), and
            # a re-save with different grouping interleaves the old
            # layout with the new one. The marker check is the
            # ownership proof: a folder we did not stamp is left alone.
            # If a "move" file mode ever becomes reachable, this wipe
            # must not run for it — the moved originals would live in
            # this tree.
            owned = (media_root / OUTPUT_DIR_MARKER).is_file()
            if owned:
                if not safe_rmtree(media_root):
                    logger.warning(
                        f"Could not fully clear {media_root}; retried "
                        f"copies may get suffixed names"
                    )
            # Mark the media folder so future scans (preview + the
            # analysis worker's input enumeration) skip it — its
            # separated / annotated copies must never be re-ingested as
            # input media. Only the media subfolder gets the marker:
            # marking the output root would make re-scans skip the whole
            # source folder when the default (source root) is used.
            #
            # Ownership rules, all load-bearing (2026-08 e2e pass):
            # - This worker is the ONLY writer of the marker. The save
            #   endpoint must never stamp it: the marker is the wipe's
            #   proof of ownership, so stamping before the check above
            #   would hand that proof to a pre-existing addaxai-media
            #   the app never created, and the wipe would delete the
            #   user's files.
            # - A pre-existing UNMARKED addaxai-media is never stamped
            #   either, or the next save would wipe it — same loss, one
            #   save later. Copies placed into such a folder keep their
            #   collision suffixes and it is never scan-skipped; both
            #   are the price of refusing to manage a tree we do not
            #   own. Best-effort.
            stamp = owned or not media_root.is_dir()
            try:
                media_root.mkdir(parents=True, exist_ok=True)
                if stamp:
                    (media_root / OUTPUT_DIR_MARKER).touch(exist_ok=True)
            except OSError as e:
                logger.warning(
                    f"Could not write output marker in {media_root}: {e}"
                )

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
        # Run-info manifest (addaxai-run-info.txt). Controlled by the Save
        # step's "Run details" checkbox; defaults to on for older frontends
        # that don't send the flag (it used to be unconditional).
        if payload.get("run_readme", True):
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

            # The two file-heavy modules copy / re-encode every file, so
            # give them a per-file callback that streams "N / M" + ETA to
            # the UI. The quick single-file writers (JSON / CSV / README)
            # don't need one; their checklist tick is feedback enough.
            file_cb = None
            if module in ("separate_folders", "annotated_copies"):
                file_cb = _make_file_progress_cb(
                    job_id,
                    loop,
                    module=module,
                    module_index=idx,
                    total_modules=total_modules,
                    active_modules=active_modules,
                )

            def _run(m: str = module, cb=file_cb) -> dict[str, Any]:
                if m == "separate_folders":
                    return separate_into_folders(
                        db,
                        project.id,
                        ctx,
                        media_threshold=media_threshold,
                        mode="copy",
                        group_by=payload.get(
                            "separate_group_by", "flat"
                        ),
                        include_empty=bool(
                            payload.get("include_empty", False)
                        ),
                        excluded_label_ids=excluded_frozen,
                        name_mode=payload.get("name_mode", "common"),
                        group_events=bool(
                            payload.get("group_events", True)
                        ),
                        species_last=bool(
                            payload.get("separate_species_last", False)
                        ),
                        # When annotated_copies also runs it re-encodes /
                        # copies the bytes straight to these destinations,
                        # so writing them here first would just be
                        # overwritten. Defer the physical write to it.
                        place_files=not (draw_bboxes or anonymise),
                        progress_cb=cb,
                    ).to_dict()
                if m == "annotated_copies":
                    return write_annotated_copies(
                        db,
                        project.id,
                        ctx,
                        media_threshold=media_threshold,
                        draw_bboxes=draw_bboxes,
                        anonymise=anonymise,
                        excluded_label_ids=excluded_frozen,
                        name_mode=payload.get("name_mode", "common"),
                        # Separation deferred its writes to us, so we own
                        # every placed file (annotated, or plain-copied when
                        # a file has no visible effect).
                        copy_unchanged=bool(
                            payload.get("separate_folders")
                        ),
                        progress_cb=cb,
                    ).to_dict()
                if m == "recognition_json":
                    return write_recognition_json(
                        db,
                        project.id,
                        output_root,
                    ).to_dict()
                if m == "csv":
                    return write_tables_csv(
                        db,
                        project.id,
                        output_root,
                    ).to_dict()
                if m == "xlsx":
                    return write_tables_xlsx(
                        db,
                        project.id,
                        output_root,
                    ).to_dict()
                if m == "run_readme":
                    return write_run_readme(
                        db,
                        project.id,
                        output_root,
                        media_threshold=media_threshold,
                    ).to_dict()
                raise ValueError(f"Unknown module: {m}")

            # Start/done lines with elapsed time and process RSS. These
            # are the only trace left when a module kills the process
            # (Wayne's 2026-08 report: three deaths in this loop with
            # nothing in the log naming the module or the memory curve).
            logger.info(
                f"save_outputs: module={module} start rss_mb={_rss_for_log()}"
            )
            module_started = time.monotonic()
            module_result = await loop.run_in_executor(None, _run)
            logger.info(
                f"save_outputs: module={module} done "
                f"elapsed_s={time.monotonic() - module_started:.1f} "
                f"rss_mb={_rss_for_log()}"
            )
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
        # Persist the message on the job row, not only on the live
        # WebSocket: after a reload or restart the failed run's error
        # text is all the user (and a diagnostics bundle) has.
        try:
            from app.api.schemas.job import JobUpdate

            job_crud.update_job(
                db, job_id, JobUpdate(status="failed", error=str(e))
            )
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
    # No "(idx / total)" suffix: it read like a file count but was a stage
    # count. The checklist already shows stage progress; the per-file
    # callback owns the "N / M files" line for the heavy stages.
    await ws_manager.send_progress(
        job_id,
        status,
        progress,
        data={
            "current_module": module,
            "module_index": module_index,
            "total_modules": total_modules,
            "modules": active_modules,
        },
    )


def _make_file_progress_cb(
    job_id: str,
    loop: asyncio.AbstractEventLoop,
    *,
    module: str,
    module_index: int,
    total_modules: int,
    active_modules: list[str],
) -> Callable[[int, int], None]:
    """Per-file progress relay for one heavy module.

    The module runs in an executor thread and calls this synchronously
    per file. It throttles (at most every ~1% and every 250ms, always
    the final tick) and schedules the async WebSocket send back on the
    event loop via ``run_coroutine_threadsafe``. The overall bar fraction
    blends completed stages with the fraction of the current stage's
    files, so it advances *within* a long stage instead of only between
    stages.
    """
    label = _MODULE_LABELS.get(module, module)
    state = {"done": -1, "t": 0.0}

    def cb(done: int, total: int) -> None:
        now = time.monotonic()
        step = max(1, total // 100)
        is_last = done >= total
        if (
            not is_last
            and done - state["done"] < step
            and now - state["t"] < 0.25
        ):
            return
        state["done"] = done
        state["t"] = now

        file_frac = done / total if total > 0 else 1.0
        overall = (
            (module_index + file_frac) / total_modules
            if total_modules > 0
            else 0.0
        )

        coro = ws_manager.send_progress(
            job_id,
            f"{label} ({done:,} / {total:,})",
            overall,
            # Within-module fraction. The save modal's bar renders this
            # one, so it matches the "(N / M)" text beside it instead of
            # sitting near zero while the first (and longest) module
            # walks its files.
            phase_progress=file_frac,
            data={
                "current_module": module,
                "module_index": module_index,
                "total_modules": total_modules,
                "modules": active_modules,
                "file_index": done,
                "file_total": total,
            },
        )
        asyncio.run_coroutine_threadsafe(coro, loop)

    return cb
