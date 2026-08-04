"""
Detection and classification worker for deployment analysis jobs.

Following DEVELOPERS.md principles:
- Crash early if configuration invalid
- Explicit error handling
- Type hints everywhere

Updated to use new ML pipeline (detection → classification).

Created by Claude Code on 2026-01-04
"""

import asyncio
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from app.api.crud import deployment as deployment_crud
from app.api.crud import deployment_queue as queue_crud
from app.api.crud import event as event_crud
from app.api.crud import job as job_crud
from app.api.crud import project as project_crud
from app.core.job_cancellation import JobCancelledError, clear_cancel
from app.core.logging_config import get_logger
from app.core.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.detection import MD_OUTPUT_CONFIDENCE_THRESHOLD
from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.custom_classification_model import CustomClassificationModel
from app.ml.inference.megadetector import MegaDetectorV1000
from app.ml.json_pipeline import merge_json_files, run_classification_on_json
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage
from app.models import Deployment
from app.services.folder_scanner import prune_unscannable_dirs
from app.utils.fs_hidden import mkdir_hidden_addaxai

logger = get_logger(__name__)


async def _process_batch_job(job_id: str, project_id: str, queue_entry_ids: list[str], db) -> None:
    """
    Process multiple queue entries sequentially within one job.

    Sends progress updates for the overall batch and each individual deployment.
    """
    logger.info(f"Batch job {job_id} starting")

    # SEND IMMEDIATE PROGRESS MESSAGE BEFORE ANY WORK
    # This ensures late-connecting WebSockets receive at least one early message
    await ws_manager.send_progress(job_id, "Job started, preparing...", 0.0)

    total_entries = len(queue_entry_ids)
    logger.info(f"Batch job {job_id}: Processing {total_entries} queue entries sequentially")

    # Update job status to running
    job_crud.update_job_status(db, job_id, "running")

    # Get project configuration (same for all entries in batch)
    project = project_crud.get_project(db, project_id)
    if not project:
        raise ValueError(f"Project not found: {project_id}")

    detection_model_id = project.detection_model_id
    classification_model_id = project.classification_model_id

    logger.info(
        f"Batch job {job_id}: Using models - detection={detection_model_id}, "
        f"classification={classification_model_id or 'None'}"
    )

    # Initialize ML infrastructure (once for all deployments)
    await ws_manager.send_progress(job_id, "Initializing ML models...", 0.01)

    manifest_manager = ManifestManager()
    env_manager = EnvironmentManager()
    model_storage = ModelStorage()

    # Load detection model
    det_manifest = manifest_manager.get_model(detection_model_id)
    det_model_path = model_storage.get_model_file(det_manifest)
    detection_model = MegaDetectorV1000(det_model_path, env_manager)

    # Load classification model (if configured)
    classification_model = None
    full_image_cls = False
    # Bound unconditionally, like its two neighbours above. It used to be
    # set only inside the branch below, which was survivable only because
    # every reader wrote `if classification_model_id and cls_model_dir`
    # and Python short-circuits before touching it. Passing it as a plain
    # argument evaluates it every time, and a detection-only run then died
    # at postprocessing with UnboundLocalError.
    cls_model_dir = None
    if classification_model_id:
        cls_manifest = manifest_manager.get_model(classification_model_id)
        cls_model_path = model_storage.get_model_file(cls_manifest)
        cls_model_dir = model_storage.get_model_path(cls_manifest)
        env_name = cls_manifest.env
        full_image_cls = bool(getattr(cls_manifest, "full_image_cls", False))

        # Check for custom inference.py script
        inference_script = cls_model_dir / "inference.py"
        if not inference_script.exists():
            error_msg = (
                f"Custom inference script not found: {inference_script}\n"
                f"Model developers must provide inference.py in their HuggingFace repo."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Use custom classification model with subprocess isolation
        logger.info(
            f"Loading custom classification model: {classification_model_id} (env: {env_name})"
        )
        classification_model = CustomClassificationModel(
            cls_model_dir, cls_model_path, env_name, env_manager
        )

    total_detections = 0
    total_files = 0
    # Reset per-iteration; only non-None if the current deployment row
    # was already created when cancel hit, so we know what to roll back.
    deployment: Deployment | None = None

    try:
        for idx, entry_id in enumerate(queue_entry_ids, start=1):
            deployment = None  # reset: each iteration creates its own
            # Get queue entry
            entry = queue_crud.get_queue_entry(db, entry_id)
            if not entry:
                logger.error(f"Queue entry {entry_id} not found, skipping")
                continue

            folder_path = Path(entry.folder_path)
            datetime_offset_seconds = entry.datetime_offset_seconds or 0
            use_file_mtime_fallback = entry.use_file_mtime_fallback
            if not folder_path.exists():
                error_msg = f"Folder not found: {folder_path}"
                logger.error(error_msg)
                queue_crud.update_queue_status(db, entry_id, status="failed", error=error_msg)
                continue

            # Progress range for this deployment
            progress_start = (idx - 1) / total_entries
            progress_end = idx / total_entries
            progress_range = progress_end - progress_start

            logger.info(
                f"Batch job {job_id}: Processing entry {idx}/{total_entries} - {folder_path}"
            )

            # Use pre-scanned counts from database if available
            # This allows us to send initial progress immediately without scanning
            if entry.video_count > 0 or entry.image_count > 0:
                logger.info(
                    f"Using pre-scanned counts: "
                    f"{entry.video_count} videos, "
                    f"{entry.image_count} images"
                )

                announced_videos, announced_images = _announced_media_counts(
                    project.media_filter, entry.video_count, entry.image_count
                )

                # Send initial progress IMMEDIATELY with pre-scanned counts
                logger.info("Sending initial progress with deployment context")
                await ws_manager.send_progress(
                    job_id,
                    "",  # Empty message, deployment header will show this info
                    progress_start,
                    phase="init",
                    phase_progress=0.0,
                    data={
                        "deployment_index": idx,
                        "total_deployments": total_entries,
                        "video_count": announced_videos,
                        "image_count": announced_images,
                        "has_classifier": classification_model is not None,
                        "has_embedding": bool(project.embedding_model_id),
                    },
                )
                logger.info("Initial progress sent, now scanning for file paths")

                # Now scan folder for actual file paths (needed for processing)
                video_files = scan_folder_for_videos(folder_path)
                image_files = scan_folder_for_images(folder_path)
            else:
                # Legacy path: no counts in database, scan first
                logger.info("No counts in database, scanning folder (legacy entry)")
                video_files = scan_folder_for_videos(folder_path)
                image_files = scan_folder_for_images(folder_path)
                logger.info("Folder scan complete")

                # Update database with scanned counts. These are what is on
                # disk, deliberately un-filtered: the step-1 caption shows
                # them, and they are the user's only clue that (say) videos
                # exist in a folder a filter is about to ignore.
                queue_crud.update_queue_counts(
                    db, entry_id, video_count=len(video_files), image_count=len(image_files)
                )

                announced_videos, announced_images = _announced_media_counts(
                    project.media_filter, len(video_files), len(image_files)
                )

                # Send initial progress with scanned counts
                logger.info("Sending initial progress with deployment context (legacy path)")
                await ws_manager.send_progress(
                    job_id,
                    "",  # Empty message, deployment header will show this info
                    progress_start,
                    phase="init",
                    phase_progress=0.0,
                    data={
                        "deployment_index": idx,
                        "total_deployments": total_entries,
                        "video_count": announced_videos,
                        "image_count": announced_images,
                        "has_classifier": classification_model is not None,
                        "has_embedding": bool(project.embedding_model_id),
                    },
                )
                logger.info("Initial progress sent")

            logger.info(
                f"Found {len(video_files)} videos and {len(image_files)} images in {folder_path}"
            )

            # Apply the media filter AFTER the scan: the queue counts above
            # stay true to disk, and we learn exactly which files were
            # dropped so they can be recorded rather than vanishing. Every
            # downstream phase is already gated on these lists being
            # non-empty, so emptying one skips that half of the pipeline —
            # the same path a photo-only folder takes.
            media_filter_skipped: list[Path] = []
            if not media_filter_allows(project.media_filter, "video"):
                media_filter_skipped.extend(video_files)
                video_files = []
            if not media_filter_allows(project.media_filter, "image"):
                media_filter_skipped.extend(image_files)
                image_files = []
            if media_filter_skipped:
                logger.info(
                    f"Media filter '{project.media_filter}': skipping "
                    f"{len(media_filter_skipped)} file(s) in {folder_path}"
                )

            if not video_files and not image_files:
                # Two different situations, and only one is the user's
                # mistake. If the filter dropped files there WAS media here,
                # so silently reporting success would be a lie; fail loudly.
                # An empty folder is not an error and keeps its old
                # behaviour. Other entries in the batch are unaffected.
                if media_filter_skipped:
                    error_msg = (
                        f"Nothing to analyse: 'Media to analyse' is set to "
                        f"'{project.media_filter}', and the folder contains "
                        f"only the other kind "
                        f"({len(media_filter_skipped)} file(s) skipped)."
                    )
                    logger.error(error_msg)
                    queue_crud.update_queue_status(
                        db, entry_id, status="failed", error=error_msg
                    )
                    continue
                logger.warning(f"No images or videos found in {folder_path}, skipping")
                queue_crud.update_queue_status(db, entry_id, status="completed")
                continue

            total_files += len(video_files) + len(image_files)

            # Create deployment (carry notes and tags from the queue entry).
            # site_id may be None for deployment-agnostic batches.
            deployment = create_deployment(
                db=db,
                project_id=entry.project_id,
                site_id=entry.site_id,
                folder_path=str(folder_path),
                notes=entry.notes,
                tags=entry.tags or {},
            )
            # Audit stamps: the datetime offset applied to timestamps,
            # and the classification gate this analysis runs under (the
            # project value can change later; mixed-gate projects need
            # the per-run record).
            if datetime_offset_seconds:
                deployment.datetime_offset_seconds = datetime_offset_seconds
            deployment.classification_gate_used = project.classification_gate
            db.commit()
            logger.info(f"Created deployment: {deployment.id}")

            # Create project-scoped artifacts folder
            artifacts_folder = folder_path / ".addaxai" / "projects" / project_id
            mkdir_hidden_addaxai(artifacts_folder)

            # JSON file paths
            video_json_path = artifacts_folder / "detection_video.json"
            image_json_path = artifacts_folder / "detection_image.json"
            final_json_path = artifacts_folder / "results.json"

            json_files_to_merge = []

            # Define progress callback for this specific deployment
            async def deployment_progress_callback(
                message: str,
                progress: float,
                phase: str,
                phase_progress: float,
                metrics: dict | None = None,
                *,
                _ps=progress_start,
                _pr=progress_range,
                _idx=idx,
                _vf=video_files,
                _if=image_files,
            ) -> None:
                """Forward progress updates (no prefix, header shows deployment context)"""
                overall_progress = _ps + (progress * _pr)
                data = {
                    "deployment_index": _idx,
                    "total_deployments": total_entries,
                    "video_count": len(_vf),
                    "image_count": len(_if),
                    "has_classifier": classification_model is not None,
                }
                # Extract compute_device from metrics to data level
                if metrics and "compute_device" in metrics:
                    data["compute_device"] = metrics.pop("compute_device")
                # Add remaining metrics if present
                if metrics:
                    data["metrics"] = metrics

                await ws_manager.send_progress(
                    job_id,
                    message,  # No prefix - raw message from ML model
                    overall_progress,
                    phase,
                    phase_progress,
                    data,
                )

            # ============================================================
            # PHASE 1: Video Detection (if videos exist)
            # ============================================================
            if video_files and full_image_cls:
                # Full-image classifiers skip MegaDetector. Synthesise a
                # video detection JSON with one full-frame bbox per
                # sampled frame so the classification phase has something
                # to consume, mirroring the image path below.
                logger.info(
                    f"Phase 1 (skipped): full-image classifier, "
                    f"synthesising video detection JSON for "
                    f"{len(video_files)} video(s)"
                )
                await deployment_progress_callback(
                    "Preparing videos...", 0.0, "video_detection", 0.5
                )
                from app.ml.full_image_detection import (
                    synthesize_full_image_video_json,
                )

                synthesize_full_image_video_json(
                    video_files, folder_path, video_json_path, project.video_fps
                )
                json_files_to_merge.append(video_json_path)

            elif video_files:
                logger.info(f"Phase 1: Running video detection on {len(video_files)} videos")

                # Create video detection model
                from app.ml.inference.video_detector import VideoDetectionModel

                video_detector = VideoDetectionModel(det_model_path, env_manager)

                # Create sync progress wrapper for executor thread
                loop = asyncio.get_event_loop()

                def sync_video_detection_progress(
                    message: str,
                    phase_progress: float,
                    metrics: dict | None = None,
                    *,
                    _loop=loop,
                ) -> None:
                    """Sync wrapper that schedules async callback from executor thread"""
                    if metrics:
                        metrics["unit"] = "video"
                    asyncio.run_coroutine_threadsafe(
                        deployment_progress_callback(
                            message, 0.0, "video_detection", phase_progress, metrics
                        ),
                        _loop,
                    )

                # Run video detection in executor (blocking subprocess I/O)
                await loop.run_in_executor(
                    None,
                    lambda _vd=video_detector,
                    _fp=folder_path,
                    _vjp=video_json_path,
                    _jid=job_id: _vd.detect_videos_to_json(
                        video_folder=_fp,
                        output_json=_vjp,
                        fps=project.video_fps,
                        confidence_threshold=MD_OUTPUT_CONFIDENCE_THRESHOLD,
                        image_size=project.detection_image_size,
                        augment=project.detection_augment,
                        progress_callback=sync_video_detection_progress,
                        job_id=_jid,
                    ),
                )

                json_files_to_merge.append(video_json_path)
                logger.info(f"Video detection complete: {video_json_path}")

            # Best-frame selection.
            #
            # When a classifier is configured, the classification worker
            # iterates each video itself (Phase 2 below) and writes the
            # winning JPEG as a side effect, so we have nothing to do
            # here. Without a classifier we still owe every video its one
            # representative frame, so run a pass that opens each video
            # once and fetches the frame the detection JSON already
            # picked out.
            #
            # Offloaded, and it has to be. This runs once per video and
            # on a folder of thousands it is minutes of work; a beta
            # tester's 1595-video run spent 85 minutes here during which
            # the backend answered no HTTP requests at all, because this
            # call sat on the event loop. It looked like a hang rather
            # than a slow step, since the websocket that would have said
            # otherwise was blocked behind it.
            if (
                video_files
                and video_json_path.exists()
                and classification_model is None
            ):
                loop = asyncio.get_running_loop()
                frame_selection_started = time.monotonic()

                def sync_frame_selection_progress(
                    done: int,
                    total: int,
                    *,
                    _loop=loop,
                    _t0=frame_selection_started,
                ) -> None:
                    """Marshal progress from the executor thread to the loop.

                    Every other phase gets its elapsed / remaining / rate
                    by parsing a subprocess's tqdm output. This one is a
                    plain Python loop, so it does the same arithmetic
                    itself and emits the same tqdm time format the
                    frontend already parses.

                    No `compute_device`. This step is CPU-only while the
                    phases around it run on the GPU, and a lone "CPU" row
                    reads as something being wrong rather than as the
                    plain fact that decoding video is not GPU work.
                    """
                    elapsed = time.monotonic() - _t0
                    metrics = {"current": done, "total": total, "unit": "video"}
                    if done and elapsed > 0:
                        rate = done / elapsed
                        metrics["rate"] = rate
                        metrics["elapsed"] = _tqdm_time(elapsed)
                        metrics["remaining"] = _tqdm_time((total - done) / rate)
                    asyncio.run_coroutine_threadsafe(
                        deployment_progress_callback(
                            f"Selecting video frames: {done}/{total}",
                            0.0,
                            "video_frame_selection",
                            (done / total) if total else 1.0,
                            metrics,
                        ),
                        _loop,
                    )

                try:
                    from app.ml.best_frame import select_best_frames_streaming

                    await asyncio.to_thread(
                        select_best_frames_streaming,
                        video_json_path,
                        deployment_folder=folder_path,
                        output_base=artifacts_folder / "video_frames",
                        progress_callback=sync_frame_selection_progress,
                        job_id=job_id,
                    )
                    logger.info("Best frame selection complete (no-classifier path)")
                except JobCancelledError:
                    # Now that this step is cancellable, the cancel has to
                    # survive the except below, which would otherwise
                    # swallow it and let the run continue as if the user
                    # never clicked.
                    raise
                except Exception as e:
                    logger.error(
                        f"Best frame selection failed: {e}", exc_info=True
                    )
                    # Non-fatal — best_frame_path on the video File will
                    # be None and the UI falls back gracefully.

            # ============================================================
            # PHASE 2: Video Classification (if videos + classifier)
            # ============================================================
            logger.debug(
                f"Phase 2 check: video_files={len(video_files) if video_files else 0}, "
                f"classification_model={classification_model is not None}, "
                f"video_json_exists={video_json_path.exists()}"
            )

            if video_files and classification_model and video_json_path.exists():
                logger.info("Phase 2: Running video classification")

                # Progress wrapper for video classification phase
                async def video_classification_progress(
                    message: str, phase_progress: float, metrics: dict | None = None
                ) -> None:
                    # Override unit to be more descriptive
                    if metrics:
                        metrics["unit"] = "animal"
                    await deployment_progress_callback(
                        message,  # Raw tqdm output
                        0.0,
                        "video_classification",
                        phase_progress,
                        metrics,
                    )

                # Run classification on video detections
                # (This will update video_json_path in-place)
                await run_classification_on_json(
                    json_path=video_json_path,
                    classification_model=classification_model,
                    deployment_folder=folder_path,
                    batch_size=project.classification_batch_size,
                    classification_gate=project.classification_gate,
                    progress_callback=video_classification_progress,
                    classification_model_dir=cls_model_dir if classification_model_id else None,
                    best_frame_output_base=artifacts_folder / "video_frames",
                    job_id=job_id,
                )

                logger.info("Video classification complete")

            # ============================================================
            # PHASE 3: Image Detection (if images exist)
            #
            # Full-image classifiers skip MegaDetector entirely. Instead,
            # we synthesise a detection JSON with one full-image bbox per
            # image so the classification phase has something to consume.
            # ============================================================
            if image_files and full_image_cls:
                logger.info(
                    f"Phase 3 (skipped): full-image classifier — "
                    f"synthesising detection JSON for "
                    f"{len(image_files)} image(s)"
                )
                await deployment_progress_callback(
                    "Preparing images...", 0.0, "image_detection", 0.5
                )
                from app.ml.full_image_detection import synthesize_full_image_json

                synthesize_full_image_json(
                    image_files, folder_path, image_json_path
                )
                json_files_to_merge.append(image_json_path)

            elif image_files:
                logger.info(f"Phase 3: Running image detection on {len(image_files)} images")

                # Create synchronous progress wrapper for executor
                loop = asyncio.get_event_loop()

                def sync_image_detection_progress(
                    message: str,
                    phase_progress: float,
                    metrics: dict | None = None,
                    *,
                    _loop=loop,
                ) -> None:
                    """Sync wrapper that schedules async callback"""
                    # Override unit to be more descriptive
                    if metrics:
                        metrics["unit"] = "image"
                    asyncio.run_coroutine_threadsafe(
                        deployment_progress_callback(
                            message,  # Raw tqdm output
                            0.0,
                            "image_detection",
                            phase_progress,
                            metrics,
                        ),
                        _loop,
                    )

                # Run MegaDetector on images
                image_json_path = await loop.run_in_executor(
                    None,
                    lambda _if=image_files,
                    _fp=folder_path,
                    _ijp=image_json_path,
                    _bs=project.detection_batch_size,
                    _jid=job_id: detection_model.detect_to_json(
                        image_paths=_if,
                        deployment_folder=_fp,
                        confidence_threshold=MD_OUTPUT_CONFIDENCE_THRESHOLD,
                        batch_size=_bs,
                        image_size=project.detection_image_size,
                        augment=project.detection_augment,
                        progress_callback=sync_image_detection_progress,
                        output_path=_ijp,
                        job_id=_jid,
                    ),
                )

                json_files_to_merge.append(image_json_path)

                logger.info(f"Image detection complete: {image_json_path}")

            # ============================================================
            # PHASE 4: Image Classification (if images + classifier)
            # ============================================================
            if image_files and classification_model and image_json_path.exists():
                logger.info("Phase 4: Running image classification")

                # Progress wrapper for image classification phase
                async def image_classification_progress(
                    message: str, phase_progress: float, metrics: dict | None = None
                ) -> None:
                    await deployment_progress_callback(
                        message,  # Raw tqdm output
                        0.0,
                        "image_classification",
                        phase_progress,
                        metrics,
                    )

                # Run classification on image detections
                await run_classification_on_json(
                    json_path=image_json_path,
                    classification_model=classification_model,
                    deployment_folder=folder_path,
                    batch_size=project.classification_batch_size,
                    classification_gate=project.classification_gate,
                    progress_callback=image_classification_progress,
                    classification_model_dir=cls_model_dir if classification_model_id else None,
                    best_frame_output_base=artifacts_folder / "video_frames",
                    job_id=job_id,
                )

                logger.info("Image classification complete")

            # ============================================================
            # PHASE 5: Merge JSONs
            # ============================================================
            if json_files_to_merge:
                logger.info(f"Phase 5: Merging {len(json_files_to_merge)} JSON files")
                await deployment_progress_callback("Merging results...", 0.0, "saving", 0.5)

                # Offloaded: merge reads/writes large JSON for the whole
                # deployment. Running it on the event loop blocks the backend
                # (see SIMON_FEEDBACK B2/B3/B4); to_thread keeps it responsive.
                await asyncio.to_thread(
                    merge_json_files,
                    json_files_to_merge,
                    final_json_path,
                    deployment.id,
                    detection_model_id=detection_model_id,
                    classification_model_id=classification_model_id,
                )

            # Trim classifications to top-5 and prune unused categories
            if final_json_path.exists() and classification_model:
                import json as _json

                from app.ml.json_utils import trim_classification_results

                with open(final_json_path) as f:
                    trimmed_data = _json.load(f)
                removed = trim_classification_results(trimmed_data)
                if removed > 0:
                    with open(final_json_path, "w") as f:
                        _json.dump(trimmed_data, f, indent=2)
                    logger.info(
                        f"Trimmed classifications: removed {removed} "
                        f"unused class IDs"
                    )

            # ============================================================
            # PRE-PHASE 6: Populate taxonomy (must exist before DB load)
            # ============================================================
            from app.ml.taxonomy_db import (
                batch_resolve_taxonomy_ids,
                ensure_builtin_labels,
                link_detections_to_taxonomy,
                populate_taxonomy_from_csv,
            )

            builtin_taxonomy_ids = ensure_builtin_labels(db)

            taxonomy_name_to_id: dict[str, tuple[str, str | None]] = {}
            if classification_model_id:
                try:
                    taxonomy_csv = cls_model_dir / "taxonomy.csv"
                    if taxonomy_csv.exists():
                        populate_taxonomy_from_csv(
                            classification_model_id, taxonomy_csv, db
                        )
                except Exception as e:
                    logger.warning(f"Failed to populate taxonomy DB: {e}")

                # Pre-resolve all class names from the JSON
                if final_json_path.exists():
                    import json as _json

                    with open(final_json_path) as _f:
                        _results_for_resolve = _json.load(_f)
                    class_names_list = list(
                        _results_for_resolve.get(
                            "classification_categories", {}
                        ).values()
                    )
                    if class_names_list:
                        taxonomy_name_to_id = batch_resolve_taxonomy_ids(
                            class_names_list,
                            classification_model_id,
                            project_id,
                            db,
                        )

            # ============================================================
            # PHASE 6: Load to Database
            # ============================================================
            if final_json_path.exists():
                logger.info("Phase 6: Loading results to database")
                await deployment_progress_callback("Loading to database...", 0.0, "saving", 0.75)

                from app.ml.json_pipeline import (
                    load_json_to_database_owned_session,
                )

                load_loop = asyncio.get_running_loop()
                load_started = time.monotonic()

                def sync_db_load_progress(
                    done: int,
                    total: int,
                    *,
                    _loop=load_loop,
                    _t0=load_started,
                ) -> None:
                    """Marshal progress from the executor thread to the loop."""
                    elapsed = time.monotonic() - _t0
                    metrics = {"current": done, "total": total, "unit": "file"}
                    if done and elapsed > 0:
                        rate = done / elapsed
                        metrics["rate"] = rate
                        metrics["elapsed"] = _tqdm_time(elapsed)
                        metrics["remaining"] = _tqdm_time((total - done) / rate)
                    asyncio.run_coroutine_threadsafe(
                        deployment_progress_callback(
                            f"Saving to database: {done}/{total}",
                            0.0,
                            "saving",
                            (done / total) if total else 1.0,
                            metrics,
                        ),
                        _loop,
                    )

                # Offloaded to a thread with its own DB session: the per-file
                # insert loop over the whole deployment is the heavy blocker of
                # the save phase. Running it on the event loop made the backend
                # hang (delete, add-to-queue) on large runs (SIMON_FEEDBACK
                # B2/B3/B4). Taxonomy is already committed above and passed in as
                # plain dicts, so the thread session only writes files/detections.
                result = await asyncio.to_thread(
                    load_json_to_database_owned_session,
                    final_json_path,
                    deployment.id,
                    folder_path,
                    job_id,
                    artifacts_folder,
                    taxonomy_name_to_id,
                    builtin_taxonomy_ids,
                    datetime_offset_seconds,
                    use_file_mtime_fallback=use_file_mtime_fallback,
                    progress_callback=sync_db_load_progress,
                )
                # The insert happened on another session; drop any stale state
                # so the main session's later queries (taxonomy link,
                # postprocessing) see the just-committed files and detections.
                db.expire_all()

                total_detections += result.total_detections
                logger.info(f"Database load complete: {result.total_detections} detections")

                # Files without a capture timestamp are now ingested
                # (data-agnostic) with captured_at_local=NULL rather than
                # dropped. We still record them so the UI can surface how
                # many lack a date and exclude them from time-based stats.
                #
                # Failed-video entries from MegaDetector's process_video
                # (corrupt file, unsupported codec, etc.) share the same
                # warnings table so the user sees both classes of issue
                # in one place. Their reason string comes straight from
                # process_video so the cause is visible.
                warning_entries: list[dict] = []
                if media_filter_skipped:
                    # Deliberate, not a fault — but the files are absent from
                    # every count and export, so the deployment has to say so.
                    # Same per-file shape as the entries below, which is what
                    # lets the UI group them into "N files: skipped…".
                    warning_entries.extend(
                        {
                            "type": "skipped_by_media_filter",
                            "path": str(p),
                        }
                        for p in media_filter_skipped
                    )
                if result.skipped_missing_timestamp:
                    logger.info(
                        f"{len(result.skipped_missing_timestamp)} file(s) "
                        "have no capture timestamp; ingested with no date"
                    )
                    warning_entries.extend(
                        {"type": "missing_timestamp", "path": p}
                        for p in result.skipped_missing_timestamp
                    )
                if result.skipped_video_failures:
                    logger.warning(
                        f"Skipped {len(result.skipped_video_failures)} "
                        "video(s) that MegaDetector could not decode"
                    )
                    warning_entries.extend(
                        {
                            "type": "video_processing_failure",
                            "path": f["file"],
                            "reason": f["reason"],
                        }
                        for f in result.skipped_video_failures
                    )
                if warning_entries:
                    queue_crud.update_queue_warnings(
                        db,
                        entry_id,
                        json.dumps(warning_entries),
                    )
                    # Mirror onto the deployment so the user can still see
                    # what was skipped after the queue row is cleaned up.
                    # Queue entries are ephemeral; the deployment is the
                    # durable record of this run.
                    deployment.warnings = warning_entries
                    db.commit()

                # Defensive fallback: link any detections that weren't
                # resolved inline (should be a no-op)
                try:
                    link_detections_to_taxonomy(project_id, db)
                except Exception as e:
                    logger.warning(f"Failed to link detections to taxonomy: {e}")

            # ============================================================
            # PHASE 7: Postprocessing (exclusion + rollup + smoothing)
            # This is the single code path for all label processing.
            # Phase 6 stores raw classifier labels; Phase 7 applies
            # exclusion, rollup, and smoothing based on project settings.
            # ============================================================
            if final_json_path.exists():
                logger.info("Phase 7: Running postprocessing")

                # Its own progress row. Everything before this point is
                # "saving"; this is label exclusion, geofencing, taxonomic
                # rollup and smoothing, which on a large deployment is
                # minutes of work of a different kind. Sharing the saving
                # row meant that row read 100% while this ran.
                pp_loop = asyncio.get_running_loop()
                pp_started = time.monotonic()

                def sync_postprocessing_progress(
                    done: int,
                    total: int,
                    *,
                    _loop=pp_loop,
                    _t0=pp_started,
                ) -> None:
                    """Marshal progress from the executor thread to the loop."""
                    elapsed = time.monotonic() - _t0
                    metrics = {"current": done, "total": total, "unit": "file"}
                    if done and elapsed > 0:
                        rate = done / elapsed
                        metrics["rate"] = rate
                        metrics["elapsed"] = _tqdm_time(elapsed)
                        metrics["remaining"] = _tqdm_time((total - done) / rate)
                    asyncio.run_coroutine_threadsafe(
                        deployment_progress_callback(
                            f"Refining results: {done}/{total}",
                            0.0,
                            "postprocessing",
                            (done / total) if total else 1.0,
                            metrics,
                        ),
                        _loop,
                    )

                await deployment_progress_callback(
                    "Refining results...", 0.0, "postprocessing", 0.0,
                )
                pp_warnings: list[dict] = []
                pp_result = await asyncio.to_thread(
                    _run_postprocessing_owned_session,
                    deployment.id,
                    project_id,
                    final_json_path,
                    folder_path,
                    classification_model_id,
                    cls_model_dir,
                    taxonomy_name_to_id,
                    job_id=job_id,
                    progress_callback=sync_postprocessing_progress,
                    warnings=pp_warnings,
                )
                # The thread committed through its own session, so rows
                # this one already loaded (project, deployment) are stale.
                db.expire_all()
                logger.info(
                    f"Postprocessing complete: {pp_result.get('updated', 0)} updated"
                )

                # Smoothing can fail without failing the run (it times out
                # and the results fall back to rollup-only). Append rather
                # than replace: phase 6 already recorded skipped files on
                # the same rows, and this must not wipe them.
                if pp_warnings:
                    existing = list(deployment.warnings or [])
                    combined = existing + pp_warnings
                    queue_crud.update_queue_warnings(
                        db, entry_id, json.dumps(combined)
                    )
                    deployment.warnings = combined
                    db.commit()

            # ============================================================
            # PHASE 8: Embedding (DINOv2) — fatal if configured
            # ============================================================
            embedding_model_id = project.embedding_model_id
            if embedding_model_id and final_json_path.exists():
                logger.info(f"Phase 8: Computing embeddings with {embedding_model_id}")
                await deployment_progress_callback("Computing embeddings...", 0.0, "embedding", 0.0)

                emb_manifest = manifest_manager.get_model(embedding_model_id)
                emb_model_path = model_storage.get_model_file(emb_manifest)

                from app.ml.embedding_utils import build_embedding_input, save_embeddings_to_db
                from app.ml.inference.embedding_model import EmbeddingModel

                embedding_model = EmbeddingModel(emb_model_path, emb_manifest, env_manager)

                input_data = build_embedding_input(
                    deployment.id,
                    db,
                    min_confidence=project.classification_gate,
                )
                embedding_input_json = artifacts_folder / "embedding_input.json"
                embedding_output_npz = artifacts_folder / "embeddings.npz"

                import json as _json

                with open(embedding_input_json, "w") as f:
                    _json.dump(input_data, f)

                # Progress wrapper for embedding phase
                loop = asyncio.get_event_loop()

                def sync_embedding_progress(
                    message: str, phase_progress: float, metrics: dict | None = None, *, _loop=loop
                ) -> None:
                    """Sync wrapper that schedules async callback from executor thread."""
                    if metrics:
                        metrics["unit"] = "crop"
                    asyncio.run_coroutine_threadsafe(
                        deployment_progress_callback(
                            message,
                            0.0,
                            "embedding",
                            phase_progress,
                            metrics,
                        ),
                        _loop,
                    )

                # Run embedding subprocess in executor (blocking I/O)
                embedded_count = await loop.run_in_executor(
                    None,
                    lambda _em=embedding_model,
                    _eij=embedding_input_json,
                    _eon=embedding_output_npz,
                    _bs=project.embedding_batch_size,
                    _jid=job_id: _em.compute_embeddings(
                        _eij, _eon, _bs, sync_embedding_progress, job_id=_jid,
                    ),
                )

                # Save embeddings to database
                if embedding_output_npz.exists():
                    save_embeddings_to_db(
                        embedding_output_npz,
                        job_id,
                        embedding_model_id,
                        emb_manifest.embedding_dim,
                        db,
                    )

                # Clean up intermediate files
                embedding_input_json.unlink(missing_ok=True)
                embedding_output_npz.unlink(missing_ok=True)

                logger.info(f"Embedding complete: {embedded_count} detections embedded")

            # Clean up intermediate JSONs (only results.json is needed at runtime)
            for intermediate in [video_json_path, image_json_path]:
                if intermediate != final_json_path and intermediate.exists():
                    intermediate.unlink()
                    logger.debug(f"Cleaned up intermediate: {intermediate.name}")

            # Send final progress update before completion
            await deployment_progress_callback("Complete", 1.0, "finalize", 1.0)

            # Update queue entry with deployment ID
            queue_crud.update_queue_status(
                db, entry_id, status="completed", deployment_id=deployment.id
            )

            logger.info(
                f"Batch job {job_id}: Completed entry {idx}/{total_entries} - "
                f"{result.total_detections} detections"
            )

        # Update postprocessing settings hash
        from app.ml.postprocessing import compute_postprocessing_settings_hash

        project.postprocessing_settings_hash = compute_postprocessing_settings_hash(project)
        db.commit()

        # Auto-generate events for the project
        event_count = event_crud.generate_events_for_project(db, project_id)
        logger.info(f"Batch job {job_id}: Auto-generated {event_count} events")

        # Refresh SQLite query planner statistics after bulk inserts
        db.execute(text("ANALYZE"))
        db.commit()

        # Mark job as completed
        job_crud.update_job_status(db, job_id, "completed")

        await ws_manager.send_complete(
            job_id=job_id,
            success=True,
            message=f"Successfully processed {total_entries} deployments",
            data={
                "deployments_processed": total_entries,
                "total_files": total_files,
                "total_detections": total_detections,
            },
        )

        logger.info(
            f"Batch job {job_id}: All {total_entries} entries completed - "
            f"{total_files} files, {total_detections} detections"
        )

    except JobCancelledError:
        # User hit Cancel. Roll back the in-flight deployment (if its
        # placeholder row exists), mark the current queue entry as
        # failed with a "Cancelled by user" note, and reset any
        # untouched-pending entries back to pending so they are
        # re-runnable. `entry_id` is bound from the for loop.
        logger.info(f"Batch job {job_id}: cancelled by user during entry {entry_id}")

        if deployment is not None:
            try:
                db.delete(deployment)
                db.commit()
                logger.info(
                    f"Rolled back placeholder deployment {deployment.id} after cancel"
                )
            except Exception as rb:
                logger.warning(f"Placeholder rollback failed after cancel: {rb}")
                db.rollback()

        # Put the in-flight entry and every remaining "processing"
        # entry back to "pending" so they're indistinguishable from
        # entries that never started. The user can re-run them by
        # hitting Run queue again.
        for other_id in queue_entry_ids:
            other = queue_crud.get_queue_entry(db, other_id)
            if other and other.status == "processing":
                queue_crud.update_queue_status(
                    db, other_id, status="pending", error=None
                )

        job_crud.update_job_status(db, job_id, "cancelled")
        await ws_manager.send_cancelled(job_id, "Run cancelled by user")

    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}", exc_info=True)
        job_crud.update_job_status(db, job_id, "failed")

        # Roll back the in-flight placeholder deployment, mirroring the
        # JobCancelledError handler. Without
        # this, a model crash or any other phase-1-7 exception leaks an
        # orphan Deployment row (today's date, the failed folder path,
        # 0 files) into the Deployments page. `deployment` is reset to
        # None at the top of each iteration, so a non-None value here is
        # unambiguously the placeholder for the entry that just failed.
        if deployment is not None:
            try:
                db.delete(deployment)
                db.commit()
                logger.info(
                    f"Rolled back placeholder deployment {deployment.id} after failure"
                )
            except Exception as rb:
                logger.warning(f"Placeholder rollback failed after batch failure: {rb}")
                db.rollback()

        # Mark remaining entries as failed
        for entry_id in queue_entry_ids:
            entry = queue_crud.get_queue_entry(db, entry_id)
            if entry and entry.status == "processing":
                queue_crud.update_queue_status(db, entry_id, status="failed", error=str(e))

        await ws_manager.send_error(job_id, str(e))
        raise

    finally:
        clear_cancel(job_id)


async def process_deployment_analysis(job_id: str) -> None:
    """
    Process deployment analysis job (detection + classification pipeline).

    Workflow:
    1. Get job and validate payload
    2. Get project configuration (detection + classification models)
    3. Scan folder for images
    4. Create deployment record
    5. Initialize ML pipeline with configured models
    6. Run pipeline: detection → classification
    7. Results saved to database with progress updates
    8. Update job status

    Args:
        job_id: Job ID to process

    Raises:
        Exception: If processing fails (caught and logged)
    """
    try:
        await ws_manager.send_progress(job_id, "Starting deployment analysis...", 0.0)

        # Get database session
        db = next(get_db())

        try:
            # Get job
            job = job_crud.get_job(db, job_id)
            if not job:
                raise ValueError(f"Job not found: {job_id}")

            # Parse payload
            payload = job.payload or {}
            project_id = payload.get("project_id")

            # Check if this is a batch job (multiple queue entries)
            is_batch = payload.get("is_batch_job", False)
            queue_entry_ids = payload.get("queue_entry_ids", [])

            if not (is_batch and queue_entry_ids):
                raise ValueError(
                    "Invalid job payload: expected is_batch_job=true "
                    "with queue_entry_ids"
                )

            logger.info(
                f"Job {job_id} is a batch job with "
                f"{len(queue_entry_ids)} entries"
            )
            await _process_batch_job(
                job_id, project_id, queue_entry_ids, db
            )

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        # Update job status
        try:
            db = next(get_db())
            job_crud.update_job_status(db, job_id, "failed")

            # Update queue entry if this job was from queue
            # Need to re-fetch job to get queue_entry_id
            job = job_crud.get_job(db, job_id)
            if job and job.payload:
                queue_entry_id = job.payload.get("queue_entry_id")
                if queue_entry_id:
                    queue_crud.update_queue_status(
                        db, queue_entry_id, status="failed", error=str(e)
                    )
                    logger.info(f"Updated queue entry {queue_entry_id} to failed")

            db.close()
        except Exception as cleanup_error:
            logger.error(f"Failed to update job/queue status: {cleanup_error}")

        # Send error message
        await ws_manager.send_error(job_id, str(e))


def _run_postprocessing_owned_session(
    deployment_id: str,
    project_id: str,
    json_path: Path,
    deployment_folder: Path,
    classification_model_id: str | None,
    cls_model_dir: Path | None,
    taxonomy_name_to_id: dict | None,
    job_id: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    warnings: list[dict] | None = None,
) -> dict:
    """
    Run phase 7 with a session created in THIS thread.

    Phase 7 used to be a bare synchronous call on the event loop, so the
    backend answered nothing for its whole duration: minutes on a large
    deployment, and it could not report its own progress because the loop
    that would send the websocket frame was the loop it was blocking.
    That is the same defect already fixed here for the merge and the
    database load, and it is the reason `asyncio.to_thread` is not
    optional.

    A SQLAlchemy session belongs to one thread, so this makes its own
    rather than borrowing the worker's, exactly like
    `load_json_to_database_owned_session`. The project is passed by id and
    re-loaded here for the same reason: handing an ORM object across the
    boundary would let a lazy load fire against a session owned by another
    thread.

    The caller must `db.expire_all()` afterwards, because rows this
    commits are stale in its own session.
    """
    from app.db.base import get_session_factory
    from app.ml.postprocessing import (
        run_postprocessing_for_deployment,
        update_database_from_smoothed_results,
    )
    from app.ml.taxonomic_rollup import load_taxonomy_lookup
    from app.ml.taxonomy_db import batch_resolve_taxonomy_ids
    from app.models import Project
    from app.models.label_taxonomy import LabelTaxonomy

    db = get_session_factory()()
    try:
        project = db.get(Project, project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found for postprocessing")

        smoothed = run_postprocessing_for_deployment(
            deployment_id, json_path, deployment_folder, project, db,
            job_id=job_id,
            warnings=warnings,
        )

        # Taxonomy for scientific_name formatting.
        pp_tax = None
        if classification_model_id and cls_model_dir:
            taxonomy_csv = cls_model_dir / "taxonomy.csv"
            if taxonomy_csv.exists():
                pp_tax = load_taxonomy_lookup(taxonomy_csv)

        # Resolve excluded_classes to taxonomy UUIDs.
        excluded_tax_ids: set[str] | None = None
        if project.excluded_classes:
            exc_rows = (
                db.query(LabelTaxonomy.id)
                .filter(LabelTaxonomy.name.in_(project.excluded_classes))
                .all()
            )
            excluded_tax_ids = {r[0] for r in exc_rows}

        # Re-resolve taxonomy after rollup may have added entries.
        pp_name_to_id = (
            batch_resolve_taxonomy_ids(
                list(smoothed.get("classification_categories", {}).values()),
                classification_model_id,
                project_id,
                db,
            )
            if classification_model_id
            else taxonomy_name_to_id
        )

        return update_database_from_smoothed_results(
            deployment_id, smoothed, deployment_folder, db, pp_tax,
            excluded_classes=project.excluded_classes,
            excluded_taxonomy_ids=excluded_tax_ids,
            taxonomy_name_to_id=pp_name_to_id,
            progress_callback=progress_callback,
        )
    finally:
        db.close()


def _tqdm_time(seconds: float) -> str:
    """
    Format seconds the way tqdm does: ``M:SS``, or ``H:MM:SS`` past an
    hour.

    Phases driven by a subprocess pass tqdm's own strings straight
    through, and the frontend's `tqdmTimeToSeconds` only accepts those
    two shapes. A phase that does its own timing has to match them or
    its times render as raw text.
    """
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def media_filter_allows(media_filter: str, file_type: str) -> bool:
    """Whether `file_type` ("image" / "video") is analysed under `media_filter`.

    The single definition of the rule, deliberately dumb. Two things must
    agree on it: the counts announced in the run header, and the file lists
    actually handed to the detector. If they disagreed, the header would
    promise media the run then silently skips.

    `media_filter` is "all" | "images" | "videos" (Project.media_filter).
    Anything unrecognised falls through to True: a filter we don't understand
    must not silently drop a user's files.
    """
    if media_filter == "images":
        return file_type == "image"
    if media_filter == "videos":
        return file_type == "video"
    return True


def _announced_media_counts(
    media_filter: str, video_count: int, image_count: int
) -> tuple[int, int]:
    """(videos, images) for the run header: 0 for whatever the filter excludes.

    The header is sent before the folder is scanned (that is the point — the
    user sees the totals immediately), so it cannot simply count the filtered
    lists. Running the same rule over the on-disk totals keeps the header
    honest about what this run will actually look at.
    """
    return (
        video_count if media_filter_allows(media_filter, "video") else 0,
        image_count if media_filter_allows(media_filter, "image") else 0,
    )


def scan_folder_for_images(folder_path: Path) -> list[Path]:
    """
    Scan folder for image files.

    Args:
        folder_path: Path to folder

    Returns:
        List of absolute paths to image files
    """
    image_files: list[Path] = []

    for root, dirs, files in os.walk(folder_path):
        # Skip dot-folders AND AddaxAI output folders, so a previous run's
        # separated / visualised copies are never re-ingested as input.
        dirs[:] = prune_unscannable_dirs(root, dirs)
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(file_path)

    # Sort by filename for consistent processing
    image_files.sort()

    return image_files


def scan_folder_for_videos(folder_path: Path) -> list[Path]:
    """
    Scan folder for video files.

    Args:
        folder_path: Path to folder

    Returns:
        List of absolute paths to video files
    """
    video_files: list[Path] = []

    for root, dirs, files in os.walk(folder_path):
        # Skip dot-folders AND AddaxAI output folders, so a previous run's
        # separated / visualised copies are never re-ingested as input.
        dirs[:] = prune_unscannable_dirs(root, dirs)
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(file_path)

    # Sort by filename for consistent processing
    video_files.sort()

    return video_files


def create_deployment(
    db,
    project_id: str,
    site_id: str | None,
    folder_path: str,
    notes: str | None = None,
    tags: dict[str, str] | None = None,
) -> Deployment:
    """
    Create deployment record.

    Args:
        db: Database session
        project_id: Project ID (required, every deployment belongs to a project)
        site_id: Site ID; None for deployment-agnostic batches
        folder_path: Folder path
        notes: Optional deployment notes (from queue entry)
        tags: Optional key:value metadata tags (from queue entry)

    Returns:
        Created Deployment
    """
    from app.api.schemas.deployment import DeploymentCreate

    # Use current UTC date as a placeholder. Phase 6 overwrites this with
    # the real field-deployment window derived from File.captured_at_local.
    deployment_data = DeploymentCreate(
        project_id=project_id,
        site_id=site_id,
        folder_path=folder_path,
        start_date_local=datetime.now(UTC).date(),
        notes=notes,
        tags=tags or {},
    )

    return deployment_crud.create_deployment(db, deployment_data)
