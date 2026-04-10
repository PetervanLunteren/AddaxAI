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
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from app.api.crud import deployment as deployment_crud
from app.api.crud import deployment_queue as queue_crud
from app.api.crud import event as event_crud
from app.api.crud import job as job_crud
from app.api.crud import project as project_crud
from app.core.logging_config import get_logger
from app.core.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.custom_classification_model import CustomClassificationModel
from app.ml.inference.megadetector import MegaDetectorV1000
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage
from app.models import Deployment

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
    if classification_model_id:
        cls_manifest = manifest_manager.get_model(classification_model_id)
        cls_model_path = model_storage.get_model_file(cls_manifest)
        cls_model_dir = model_storage.get_model_path(cls_manifest)
        env_name = cls_manifest.env

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

    try:
        for idx, entry_id in enumerate(queue_entry_ids, start=1):
            # Get queue entry
            entry = queue_crud.get_queue_entry(db, entry_id)
            if not entry:
                logger.error(f"Queue entry {entry_id} not found, skipping")
                continue

            folder_path = Path(entry.folder_path)
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
                        "video_count": entry.video_count,
                        "image_count": entry.image_count,
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

                # Update database with scanned counts
                queue_crud.update_queue_counts(
                    db, entry_id, video_count=len(video_files), image_count=len(image_files)
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
                        "video_count": len(video_files),
                        "image_count": len(image_files),
                        "has_classifier": classification_model is not None,
                        "has_embedding": bool(project.embedding_model_id),
                    },
                )
                logger.info("Initial progress sent")

            logger.info(
                f"Found {len(video_files)} videos and {len(image_files)} images in {folder_path}"
            )

            if not video_files and not image_files:
                logger.warning(f"No images or videos found in {folder_path}, skipping")
                queue_crud.update_queue_status(db, entry_id, status="completed")
                continue

            total_files += len(video_files) + len(image_files)

            # Validate site selection from queue entry
            if not entry.site_id:
                error_msg = f"Queue entry {entry_id} has no site selected"
                logger.error(error_msg)
                queue_crud.update_queue_status(db, entry_id, status="failed", error=error_msg)
                continue

            # Create deployment
            deployment = create_deployment(
                db=db, site_id=entry.site_id, folder_path=str(folder_path)
            )
            logger.info(f"Created deployment: {deployment.id}")

            # Create project-scoped artifacts folder
            artifacts_folder = folder_path / ".addaxai" / "projects" / project_id
            artifacts_folder.mkdir(parents=True, exist_ok=True)

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
            if video_files:
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
                    _vjp=video_json_path: _vd.detect_videos_to_json(
                        video_folder=_fp,
                        output_json=_vjp,
                        fps=project.video_fps,
                        confidence_threshold=0.1,
                        progress_callback=sync_video_detection_progress,
                    ),
                )

                json_files_to_merge.append(video_json_path)
                logger.info(f"Video detection complete: {video_json_path}")

            # Extract ALL video frames to disk (for frame-level DB records)
            if video_files and video_json_path.exists():
                try:
                    from app.ml.frame_extraction import extract_all_video_frames

                    extract_all_video_frames(
                        folder_path,
                        project.video_fps,
                        env_manager,
                        output_dir=artifacts_folder / "video_frames",
                    )
                    logger.info("Video frame extraction complete")
                except Exception as e:
                    logger.error(f"Video frame extraction failed: {e}", exc_info=True)
                    # Non-fatal — continue pipeline

            # Best frame selection (scoring only — frames already on disk)
            if video_files and video_json_path.exists():
                try:
                    from app.ml.best_frame import select_best_frames

                    select_best_frames(video_json_path, artifacts_folder / "video_frames")
                    logger.info("Best frame selection complete")
                except Exception as e:
                    logger.error(f"Best frame selection failed: {e}", exc_info=True)
                    # Non-fatal — continue pipeline

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
                    progress_callback=video_classification_progress,
                    classification_model_dir=cls_model_dir if classification_model_id else None,
                    video_frames_base_dir=artifacts_folder / "video_frames",
                )

                logger.info("Video classification complete")

            # ============================================================
            # PHASE 3: Image Detection (if images exist)
            # ============================================================
            if image_files:
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
                    _bs=project.detection_batch_size: detection_model.detect_to_json(
                        image_paths=_if,
                        deployment_folder=_fp,
                        confidence_threshold=0.1,
                        batch_size=_bs,
                        progress_callback=sync_image_detection_progress,
                        output_path=_ijp,
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
                    progress_callback=image_classification_progress,
                    classification_model_dir=cls_model_dir if classification_model_id else None,
                    video_frames_base_dir=artifacts_folder / "video_frames",
                )

                logger.info("Image classification complete")

            # ============================================================
            # PHASE 5: Merge JSONs
            # ============================================================
            if json_files_to_merge:
                logger.info(f"Phase 5: Merging {len(json_files_to_merge)} JSON files")
                await deployment_progress_callback("Merging results...", 0.0, "saving", 0.5)

                merge_json_files(
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

                from app.ml.json_pipeline import load_json_to_database

                result = load_json_to_database(
                    json_path=final_json_path,
                    deployment_id=deployment.id,
                    deployment_folder=folder_path,
                    job_id=job_id,
                    db=db,
                    artifacts_folder=artifacts_folder,
                    taxonomy_name_to_id=taxonomy_name_to_id,
                    builtin_taxonomy_ids=builtin_taxonomy_ids,
                )

                total_detections += result.total_detections
                logger.info(f"Database load complete: {result.total_detections} detections")

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
                from app.ml.postprocessing import (
                    run_postprocessing_for_deployment,
                    update_database_from_smoothed_results,
                )

                smoothed = run_postprocessing_for_deployment(
                    deployment.id, final_json_path, folder_path, project, db
                )
                # Load taxonomy for display_name formatting
                taxonomy_csv = None
                if classification_model_id and cls_model_dir:
                    _tax = cls_model_dir / "taxonomy.csv"
                    if _tax.exists():
                        taxonomy_csv = _tax
                pp_tax = None
                if taxonomy_csv and taxonomy_csv.exists():
                    from app.ml.taxonomic_rollup import load_taxonomy_lookup

                    pp_tax = load_taxonomy_lookup(taxonomy_csv)

                # Resolve excluded_classes to taxonomy UUIDs
                excluded_tax_ids: set[str] | None = None
                if project.excluded_classes:
                    from app.models.label_taxonomy import LabelTaxonomy

                    exc_rows = (
                        db.query(LabelTaxonomy.id)
                        .filter(
                            LabelTaxonomy.name.in_(
                                project.excluded_classes
                            ),
                        )
                        .all()
                    )
                    excluded_tax_ids = {r[0] for r in exc_rows}

                # Re-resolve taxonomy after rollup may have added entries
                pp_name_to_id = batch_resolve_taxonomy_ids(
                    list(
                        smoothed.get(
                            "classification_categories", {}
                        ).values()
                    ),
                    classification_model_id,
                    project_id,
                    db,
                ) if classification_model_id else taxonomy_name_to_id

                pp_result = update_database_from_smoothed_results(
                    deployment.id, smoothed, folder_path, db, pp_tax,
                    excluded_classes=project.excluded_classes,
                    excluded_taxonomy_ids=excluded_tax_ids,
                    taxonomy_name_to_id=pp_name_to_id,
                )
                logger.info(
                    f"Postprocessing complete: {pp_result.get('updated', 0)} updated"
                )

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

                input_data = build_embedding_input(deployment.id, db)
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
                    _bs=project.embedding_batch_size: _em.compute_embeddings(
                        _eij, _eon, _bs, sync_embedding_progress
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

    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}", exc_info=True)
        job_crud.update_job_status(db, job_id, "failed")

        # Mark remaining entries as failed
        for entry_id in queue_entry_ids:
            entry = queue_crud.get_queue_entry(db, entry_id)
            if entry and entry.status == "processing":
                queue_crud.update_queue_status(db, entry_id, status="failed", error=str(e))

        await ws_manager.send_error(job_id, str(e))
        raise


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
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # skip .addaxai etc.
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(file_path)

    # Sort by filename for consistent processing
    image_files.sort()

    return image_files


async def run_classification_on_json(
    json_path: Path,
    classification_model,
    deployment_folder: Path,
    batch_size: int,
    progress_callback: Callable[[str, float, dict | None], None] | None = None,
    classification_model_dir: Path | None = None,
    video_frames_base_dir: Path | None = None,
) -> None:
    """
    Run classification on detection JSON file.

    Updates JSON file in-place with classification results.

    Args:
        json_path: Path to detection JSON file
        classification_model: Classification model instance
        deployment_folder: Deployment folder for artifacts
        batch_size: Number of crops per classification batch. Resolved by the
            caller from the project's classification_batch_size override (or
            the per-pipeline default).
        progress_callback: Optional progress callback
        classification_model_dir: Path to classification model directory (for taxonomy.csv)
        video_frames_base_dir: Path to video_frames directory. If None, falls back to
            deployment_folder / ".addaxai" / "video_frames".

    Raises:
        RuntimeError: If classification fails
    """
    import json

    from app.ml.json_utils import extract_animal_detections

    logger.info("Running per-detection classification")

    # Load detection JSON
    with open(json_path) as f:
        md_results = json.load(f)

    # Extract animal detections
    animal_detections = extract_animal_detections(md_results)
    total_animals = len(animal_detections)

    if total_animals == 0:
        logger.info("No animals to classify")
        return

    # Build items list and parallel index list for result merging
    items: list[dict] = []
    indices: list[tuple[int, int]] = []

    for img_idx, det_idx, detection in animal_detections:
        img_info = md_results["images"][img_idx]
        relative_file = img_info["file"]
        file_path = (deployment_folder / relative_file).resolve()

        is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS

        # For videos: resolve to extracted frame JPEG
        if is_video:
            frame_number = detection.get("frame_number")
            if frame_number is None:
                logger.warning("Detection missing frame_number, skipping")
                continue

            _frames_base = video_frames_base_dir or (
                deployment_folder / ".addaxai" / "video_frames"
            )
            relative_video_path = file_path.relative_to(deployment_folder)
            frame_path = (
                _frames_base / relative_video_path / f"frame{frame_number:06d}.jpg"
            )
            if not frame_path.exists():
                logger.warning(f"Frame {frame_path.name} not found on disk, skipping")
                continue
            image_path = str(frame_path)
        else:
            if not file_path.exists():
                logger.warning(f"Image not found: {file_path}, skipping")
                continue
            image_path = str(file_path)

        items.append({
            "image_path": image_path,
            "bbox": detection["bbox"],
        })
        indices.append((img_idx, det_idx))

    # Debug: summarize what we built
    video_items = sum(1 for it in items if "frame" in it["image_path"])
    image_items = len(items) - video_items
    logger.info(
        f"[DEBUG] Built {len(items)} items for batch classification "
        f"({image_items} images, {video_items} video frames), "
        f"{len(indices)} indices"
    )

    if not items:
        logger.info("No valid items to classify after path resolution")
        return

    # Create sync progress wrapper for executor thread
    loop = asyncio.get_event_loop()

    def sync_cls_progress(
        message: str, phase_progress: float, metrics: dict | None = None
    ) -> None:
        """Sync wrapper that schedules async callback from executor thread"""
        if progress_callback:
            asyncio.run_coroutine_threadsafe(
                progress_callback(message, phase_progress, metrics), loop
            )

    def _run_batch_classification():
        """Synchronous batch classification (runs in executor)."""
        import time

        start_time = time.time()

        def on_progress(current: int, total: int) -> None:
            if not progress_callback:
                return
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
            rate = current / elapsed if elapsed > 0 else 0
            remaining = (total - current) / rate if rate > 0 else 0
            remaining_str = f"{int(remaining//60):02d}:{int(remaining%60):02d}"
            percent = int(100 * current / total)
            bar_length = 10
            filled = int(bar_length * current / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            raw_line = (
                f"{percent}%|{bar}| {current}/{total} "
                f"[{elapsed_str}<{remaining_str}, {rate:.2f}animal/s]"
            )
            metrics = {
                "raw_line": raw_line,
                "current": current,
                "total": total,
                "elapsed": elapsed_str,
                "remaining": remaining_str,
                "rate": rate,
                "unit": "animal",
            }
            sync_cls_progress(raw_line, current / total, metrics)

        logger.info("[DEBUG] Calling classify_detections()...")
        results, class_names, compute_device = classification_model.classify_detections(
            items, batch_size=batch_size, progress_callback=on_progress,
        )
        logger.info(
            f"[DEBUG] classify_detections() returned: "
            f"{len(results)} results, {len(class_names)} classes, device={compute_device}"
        )

        # Send compute device info (at phase_progress=1.0 to avoid
        # resetting the progress bar after classification finishes)
        if progress_callback and compute_device:
            sync_cls_progress("Classifying...", 1.0, {"compute_device": compute_device})

        # Merge results back into md_results JSON
        name_to_id = {name: class_id for class_id, name in class_names.items()}
        classified_count = 0

        for (img_idx, det_idx), result in zip(indices, results, strict=True):
            if result is None:
                continue

            # Store all results (not truncated) so label exclusion
            # can find included labels even if they rank low.
            md_results["images"][img_idx]["detections"][det_idx]["classifications"] = [
                [name_to_id[class_name], prob]
                for class_name, prob in result.all_probabilities.items()
                if class_name in name_to_id
            ]
            classified_count += 1

        # Add classification metadata to JSON
        if class_names:
            md_results["classification_categories"] = class_names

            if classification_model_dir:
                taxonomy_csv = classification_model_dir / "taxonomy.csv"
                if taxonomy_csv.exists():
                    from app.ml.json_utils import build_classification_category_descriptions

                    descriptions = build_classification_category_descriptions(
                        class_names, taxonomy_csv
                    )
                    if descriptions:
                        md_results["classification_category_descriptions"] = descriptions

        # Save updated JSON
        with open(json_path, "w") as f:
            json.dump(md_results, f, indent=2)

        logger.info(f"Classified {classified_count}/{total_animals} animals")
        logger.info(
            f"[DEBUG] Wrote updated JSON to {json_path}, "
            f"has classification_categories={bool(md_results.get('classification_categories'))}"
        )

    # Run in executor to avoid blocking event loop
    await loop.run_in_executor(None, _run_batch_classification)


def merge_json_files(
    json_files: list[Path],
    output_file: Path,
    deployment_id: str,
    detection_model_id: str | None = None,
    classification_model_id: str | None = None,
) -> None:
    """
    Merge multiple JSON files (video and image results) into single file.

    Creates a unified classification_categories mapping and renumbers all
    classification IDs to be consistent across video and image detections.

    This is necessary because video and image JSONs may have different ID
    mappings for the same label. This function unifies the mappings so all
    IDs are consistent.

    Args:
        json_files: List of JSON file paths to merge
        output_file: Output merged JSON file path
        deployment_id: Deployment ID for metadata
        detection_model_id: Detection model ID (for info section)
        classification_model_id: Classification model ID (for info section)

    Raises:
        RuntimeError: If merge fails
    """
    import json

    try:
        merged_data = {
            "images": [],
            "detection_categories": {},
            "classification_categories": {},
            "classification_category_descriptions": {},
            "info": {},
        }

        # Track unified classification mapping: label_name -> unified_id
        unified_class_mapping = {}
        next_class_id = 1

        for json_file in json_files:
            if not json_file.exists():
                logger.warning(f"JSON file not found: {json_file}")
                continue

            with open(json_file) as f:
                data = json.load(f)

            # Get classification categories from this file
            file_class_categories = data.get("classification_categories", {})

            # Build mapping from old ID to new ID for this file
            id_remapping = {}

            # For each label in this file's classification_categories
            for old_id, label_name in file_class_categories.items():
                # Check if we've seen this label before
                if label_name not in unified_class_mapping:
                    # New label - assign next available ID
                    unified_class_mapping[label_name] = str(next_class_id)
                    next_class_id += 1

                # Map old ID to unified ID
                id_remapping[old_id] = unified_class_mapping[label_name]

            # Remap classification_category_descriptions using the same ID remapping
            file_descriptions = data.get("classification_category_descriptions", {})
            for old_id, desc_str in file_descriptions.items():
                new_id = id_remapping.get(old_id, old_id)
                merged_data["classification_category_descriptions"][new_id] = desc_str

            # Renumber classification IDs in all detections from this file
            for img in data.get("images", []):
                if "detections" in img:
                    for det in img["detections"]:
                        if "classifications" in det and det["classifications"]:
                            # Renumber each classification
                            renumbered_classifications = []
                            for class_id, confidence in det["classifications"]:
                                old_id_str = str(class_id)
                                if old_id_str in id_remapping:
                                    new_id = id_remapping[old_id_str]
                                    renumbered_classifications.append([new_id, confidence])
                                else:
                                    # Unknown class ID - keep original (shouldn't happen)
                                    logger.warning(
                                        f"Unknown classification ID "
                                        f"'{class_id}' in "
                                        f"{json_file.name}, "
                                        f"keeping original"
                                    )
                                    renumbered_classifications.append([class_id, confidence])

                            det["classifications"] = renumbered_classifications

            # Merge images arrays (with renumbered IDs)
            merged_data["images"].extend(data.get("images", []))

            # Use detection_categories and info from first file
            if not merged_data["detection_categories"]:
                merged_data["detection_categories"] = data.get("detection_categories", {})
            if not merged_data["info"]:
                merged_data["info"] = data.get("info", {})

        # Build unified classification_categories (inverse of unified_class_mapping)
        merged_data["classification_categories"] = {
            class_id: label_name for label_name, class_id in unified_class_mapping.items()
        }

        # Remove empty descriptions dict
        if not merged_data["classification_category_descriptions"]:
            del merged_data["classification_category_descriptions"]

        num_labels = len(merged_data['classification_categories'])
        logger.info(
            f"Unified classification mapping: "
            f"{num_labels} labels "
            f"across {len(json_files)} JSON files"
        )

        # Add AddaxAI metadata nested under info.addaxai
        addaxai_info: dict = {
            "version": "todo-not-implemented-yet",
            "deployment_id": deployment_id,
            "classification_completion_time": (
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            ),
        }
        if detection_model_id:
            addaxai_info["detection_model"] = detection_model_id
        if classification_model_id:
            addaxai_info["classification_model"] = (
                classification_model_id
            )
        merged_data["info"]["addaxai"] = addaxai_info

        # Write merged JSON
        with open(output_file, "w") as f:
            json.dump(merged_data, f, indent=2)

        logger.info(f"Merged {len(json_files)} JSON files to {output_file}")

    except Exception as e:
        logger.error(f"JSON merge failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to merge JSON files: {e}") from e


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
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # skip .addaxai etc.
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(file_path)

    # Sort by filename for consistent processing
    video_files.sort()

    return video_files


def create_deployment(db, site_id: str, folder_path: str) -> Deployment:
    """
    Create deployment record.

    Args:
        db: Database session
        site_id: Site ID
        folder_path: Folder path

    Returns:
        Created Deployment
    """
    from app.api.schemas.deployment import DeploymentCreate

    # Use current date as start date
    deployment_data = DeploymentCreate(
        site_id=site_id,
        folder_path=folder_path,
        start_date=datetime.utcnow().date(),
    )

    return deployment_crud.create_deployment(db, deployment_data)
