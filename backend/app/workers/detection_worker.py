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
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image
from PIL.ExifTags import TAGS

from app.api.crud import deployment as deployment_crud
from app.api.crud import deployment_queue as queue_crud
from app.api.crud import job as job_crud
from app.api.crud import project as project_crud
from app.api.crud import site as site_crud
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.custom_classification_model import CustomClassificationModel
from app.ml.inference.megadetector import MegaDetectorV1000
from app.ml.json_pipeline import JSONBasedMLPipeline
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage
from app.models import Deployment

logger = get_logger(__name__)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Supported video formats (MegaDetector compatible)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mpeg", ".mpg", ".mov", ".mkv", ".flv"}


async def _process_batch_job(job_id: str, project_id: str, queue_entry_ids: list[str], db) -> None:
    """
    Process multiple queue entries sequentially within one job.

    Sends progress updates for the overall batch and each individual deployment.
    """
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

        # Detect SpeciesNet by model ID (future-proof, no dependency on manifest.type)
        is_speciesnet = "SPECIESNET" in classification_model_id.upper()

        if is_speciesnet:
            # SpeciesNet: batch processing, no inference.py needed
            logger.info(f"Loading SpeciesNet model: {classification_model_id} (env: {env_name})")
            from app.ml.inference.speciesnet_model import SpeciesNetClassificationModel

            classification_model = SpeciesNetClassificationModel(
                cls_model_dir, cls_model_path, env_name, env_manager
            )
        else:
            # Regular models: check for custom inference.py script
            inference_script = cls_model_dir / "inference.py"
            if not inference_script.exists():
                error_msg = (
                    f"Custom inference script not found: {inference_script}\n"
                    f"Model developers must provide inference.py in their HuggingFace repo."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Use custom classification model with subprocess isolation
            logger.info(f"Loading custom classification model: {classification_model_id} (env: {env_name})")
            classification_model = CustomClassificationModel(
                cls_model_dir, cls_model_path, env_name, env_manager
            )

    # Create JSON-based ML pipeline with country/state for SpeciesNet
    pipeline = JSONBasedMLPipeline(
        detection_model,
        classification_model,
        detection_model_id,
        classification_model_id,
        country_code=project.country_code,
        state_code=project.state_code,
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

            logger.info(f"Batch job {job_id}: Processing entry {idx}/{total_entries} - {folder_path}")

            # Send initial progress
            await ws_manager.send_progress(
                job_id,
                f"[{idx}/{total_entries}] Scanning {folder_path.name}...",
                progress_start
            )

            # Scan folder for images and videos (separate)
            video_files = scan_folder_for_videos(folder_path)
            image_files = scan_folder_for_images(folder_path)

            logger.info(
                f"Found {len(video_files)} videos and {len(image_files)} images in {folder_path}"
            )

            if not video_files and not image_files:
                logger.warning(f"No images or videos found in {folder_path}, skipping")
                queue_crud.update_queue_status(db, entry_id, status="completed")
                continue

            total_files += len(video_files) + len(image_files)

            # Get or create "Unknown Site"
            site = site_crud.get_or_create_unknown_site(db, project_id)

            # Create deployment
            deployment = create_deployment(db=db, site_id=site.id, folder_path=str(folder_path))
            logger.info(f"Created deployment: {deployment.id}")

            # Create artifacts folder
            artifacts_folder = folder_path / ".addaxai"
            artifacts_folder.mkdir(parents=True, exist_ok=True)

            # JSON file paths
            video_json_path = artifacts_folder / "detection_video.json"
            image_json_path = artifacts_folder / "detection_image.json"
            final_json_path = artifacts_folder / "results_with_classifications.json"

            json_files_to_merge = []

            # Define progress callback for this specific deployment
            async def deployment_progress_callback(
                message: str, progress: float, phase: str, phase_progress: float
            ) -> None:
                """Forward progress updates with deployment number prefix"""
                overall_progress = progress_start + (progress * progress_range)
                await ws_manager.send_progress(
                    job_id,
                    f"[{idx}/{total_entries}] {message}",
                    overall_progress,
                    phase,
                    phase_progress
                )

            # ============================================================
            # PHASE 1: Video Detection (if videos exist)
            # ============================================================
            if video_files:
                logger.info(f"Phase 1: Running video detection on {len(video_files)} videos")

                try:
                    # Create video detection model
                    from app.ml.inference.video_detector import VideoDetectionModel
                    video_detector = VideoDetectionModel(det_model_path, env_manager)

                    # Progress wrapper for video detection phase
                    async def video_detection_progress(message: str, phase_progress: float) -> None:
                        await deployment_progress_callback(
                            f"Video detection: {message}",
                            0.0,
                            "video_detection",
                            phase_progress
                        )

                    # Run video detection
                    await video_detector.detect_videos_to_json(
                        video_folder=folder_path,
                        output_json=video_json_path,
                        fps=project.video_fps,
                        confidence_threshold=0.1,
                        progress_callback=video_detection_progress,
                    )

                    json_files_to_merge.append(video_json_path)
                    logger.info(f"Video detection complete: {video_json_path}")

                except Exception as e:
                    logger.error(f"Video detection failed: {e}", exc_info=True)
                    # Continue with images even if video fails

            # ============================================================
            # PHASE 2: Video Classification (if videos + classifier)
            # ============================================================
            logger.debug(f"Phase 2 check: video_files={len(video_files) if video_files else 0}, "
                        f"classification_model={classification_model is not None}, "
                        f"video_json_exists={video_json_path.exists()}")

            if video_files and classification_model and video_json_path.exists():
                logger.info("Phase 2: Running video classification")

                try:
                    # Progress wrapper for video classification phase
                    async def video_classification_progress(message: str, phase_progress: float) -> None:
                        await deployment_progress_callback(
                            f"Video classification: {message}",
                            0.0,
                            "video_classification",
                            phase_progress
                        )

                    # Run classification on video detections
                    # (This will update video_json_path in-place)
                    await run_classification_on_json(
                        json_path=video_json_path,
                        classification_model=classification_model,
                        deployment_folder=folder_path,
                        country_code=project.country_code,
                        state_code=project.state_code,
                        progress_callback=video_classification_progress,
                    )

                    logger.info(f"Video classification complete")

                except Exception as e:
                    logger.error(f"Video classification failed: {e}", exc_info=True)
                    # Continue with images

            # ============================================================
            # PHASE 3: Image Detection (if images exist)
            # ============================================================
            if image_files:
                logger.info(f"Phase 3: Running image detection on {len(image_files)} images")

                try:
                    # Send initial progress update
                    await deployment_progress_callback(
                        f"Image detection: Starting detection on {len(image_files)} images...",
                        0.0,
                        "image_detection",
                        0.0
                    )

                    # Create synchronous progress wrapper for executor
                    loop = asyncio.get_event_loop()
                    def sync_image_detection_progress(message: str, phase_progress: float) -> None:
                        """Sync wrapper that schedules async callback"""
                        asyncio.run_coroutine_threadsafe(
                            deployment_progress_callback(
                                f"Image detection: {message}",
                                0.0,
                                "image_detection",
                                phase_progress
                            ),
                            loop
                        )

                    # Run MegaDetector on images
                    image_json_path = await loop.run_in_executor(
                        None,
                        lambda: detection_model.detect_to_json(
                            image_paths=image_files,
                            deployment_folder=folder_path,
                            confidence_threshold=0.1,
                            progress_callback=sync_image_detection_progress,
                        ),
                    )

                    json_files_to_merge.append(image_json_path)

                    # Send completion progress update
                    await deployment_progress_callback(
                        f"Image detection: Completed {len(image_files)} images",
                        0.0,
                        "image_detection",
                        1.0
                    )

                    logger.info(f"Image detection complete: {image_json_path}")

                except Exception as e:
                    logger.error(f"Image detection failed: {e}", exc_info=True)

            # ============================================================
            # PHASE 4: Image Classification (if images + classifier)
            # ============================================================
            if image_files and classification_model and image_json_path.exists():
                logger.info("Phase 4: Running image classification")

                try:
                    # Send initial progress update
                    await deployment_progress_callback(
                        f"Image classification: Starting classification...",
                        0.0,
                        "image_classification",
                        0.0
                    )

                    # Progress wrapper for image classification phase
                    async def image_classification_progress(message: str, phase_progress: float) -> None:
                        await deployment_progress_callback(
                            f"Image classification: {message}",
                            0.0,
                            "image_classification",
                            phase_progress
                        )

                    # Run classification on image detections
                    await run_classification_on_json(
                        json_path=image_json_path,
                        classification_model=classification_model,
                        deployment_folder=folder_path,
                        country_code=project.country_code,
                        state_code=project.state_code,
                        progress_callback=image_classification_progress,
                    )

                    # Send completion progress update
                    await deployment_progress_callback(
                        f"Image classification: Complete",
                        0.0,
                        "image_classification",
                        1.0
                    )

                    logger.info(f"Image classification complete")

                except Exception as e:
                    logger.error(f"Image classification failed: {e}", exc_info=True)

            # ============================================================
            # PHASE 5: Merge JSONs
            # ============================================================
            if json_files_to_merge:
                logger.info(f"Phase 5: Merging {len(json_files_to_merge)} JSON files")
                await deployment_progress_callback("Merging results...", 0.0, "finalize", 0.5)

                merge_json_files(json_files_to_merge, final_json_path, deployment.id)

            # ============================================================
            # PHASE 6: Load to Database
            # ============================================================
            if final_json_path.exists():
                logger.info("Phase 6: Loading results to database")
                await deployment_progress_callback("Loading to database...", 0.0, "finalize", 0.75)

                from app.ml.json_pipeline import load_json_to_database

                result = load_json_to_database(
                    json_path=final_json_path,
                    deployment_id=deployment.id,
                    deployment_folder=folder_path,
                    job_id=job_id,
                    db=db,
                )

                total_detections += result.total_detections
                logger.info(f"Database load complete: {result.total_detections} detections")

            # Send final progress update before completion
            await deployment_progress_callback("Complete", 1.0, "finalize", 1.0)

            # Update queue entry with deployment ID
            queue_crud.update_queue_status(db, entry_id, status="completed", deployment_id=deployment.id)

            logger.info(
                f"Batch job {job_id}: Completed entry {idx}/{total_entries} - "
                f"{result.total_detections} detections"
            )

        # Mark job as completed
        job_crud.update_job_status(db, job_id, "completed")

        await ws_manager.send_complete(
            job_id=job_id,
            success=True,
            message=f"Successfully processed {total_entries} deployments",
            data={
                "deployments_processed": total_entries,
                "total_files": total_files,
                "total_detections": total_detections
            }
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

            if is_batch and queue_entry_ids:
                # Process multiple queue entries sequentially
                logger.info(f"Job {job_id} is a batch job with {len(queue_entry_ids)} entries")
                await _process_batch_job(job_id, project_id, queue_entry_ids, db)
                # Don't continue with single-deployment logic
                db.close()
                return

            # Single deployment processing (original logic)
            folder_path = payload.get("folder_path")
            queue_entry_id = payload.get("queue_entry_id")  # Optional - may be None if not from queue

            if not all([project_id, folder_path]):
                raise ValueError("Invalid job payload: missing project_id or folder_path")

            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise ValueError(f"Folder not found: {folder_path}")

            # Get project configuration
            await ws_manager.send_progress(job_id, "Loading project configuration...", 0.02)
            project = project_crud.get_project(db, project_id)
            if not project:
                raise ValueError(f"Project not found: {project_id}")

            detection_model_id = project.detection_model_id
            classification_model_id = project.classification_model_id

            logger.info(
                f"Project {project.name}: detection={detection_model_id}, "
                f"classification={classification_model_id or 'None'}"
            )

            # Update job status
            job_crud.update_job_status(db, job_id, "running")

            # Scan folder for images
            await ws_manager.send_progress(job_id, "Scanning folder for images...", 0.03)
            image_files = scan_folder_for_images(folder_path)
            logger.info(f"Found {len(image_files)} images in {folder_path}")

            if not image_files:
                raise ValueError(f"No images found in {folder_path}")

            # Get or create "Unknown Site"
            await ws_manager.send_progress(job_id, "Creating deployment...", 0.04)
            site = site_crud.get_or_create_unknown_site(db, project_id)

            # Create deployment
            deployment = create_deployment(
                db=db,
                site_id=site.id,
                folder_path=str(folder_path),
            )
            logger.info(f"Created deployment: {deployment.id}")

            # Initialize ML infrastructure
            await ws_manager.send_progress(job_id, "Initializing ML models...", 0.05)

            manifest_manager = ManifestManager()
            env_manager = EnvironmentManager()
            model_storage = ModelStorage()

            # Load detection model
            det_manifest = manifest_manager.get_model(detection_model_id)
            det_model_path = model_storage.get_model_file(det_manifest)

            logger.info(f"Loading detection model: {detection_model_id}")
            detection_model = MegaDetectorV1000(det_model_path, env_manager)

            # Load classification model (if configured)
            classification_model = None
            if classification_model_id:
                cls_manifest = manifest_manager.get_model(classification_model_id)
                cls_model_path = model_storage.get_model_file(cls_manifest)
                cls_model_dir = model_storage.get_model_path(cls_manifest)
                env_name = cls_manifest.env

                # Detect SpeciesNet by model ID (future-proof, no dependency on manifest.type)
                is_speciesnet = "SPECIESNET" in classification_model_id.upper()

                if is_speciesnet:
                    # SpeciesNet: batch processing, no inference.py needed
                    logger.info(f"Loading SpeciesNet model: {classification_model_id} (env: {env_name})")
                    from app.ml.inference.speciesnet_model import SpeciesNetClassificationModel

                    classification_model = SpeciesNetClassificationModel(
                        cls_model_dir, cls_model_path, env_name, env_manager
                    )
                else:
                    # Regular models: check for custom inference.py script
                    inference_script = cls_model_dir / "inference.py"
                    if not inference_script.exists():
                        error_msg = (
                            f"Custom inference script not found: {inference_script}\n"
                            f"Model developers must provide inference.py in their HuggingFace repo."
                        )
                        logger.error(error_msg)
                        raise FileNotFoundError(error_msg)

                    # Use custom classification model with subprocess isolation
                    logger.info(f"Loading custom classification model: {classification_model_id} (env: {env_name})")
                    classification_model = CustomClassificationModel(
                        cls_model_dir, cls_model_path, env_name, env_manager
                    )

            # Create JSON-based ML pipeline with country/state for SpeciesNet
            pipeline = JSONBasedMLPipeline(
                detection_model,
                classification_model,
                detection_model_id,
                classification_model_id,
                country_code=project.country_code,
                state_code=project.state_code,
            )

            # Define progress callback wrapper
            async def progress_callback(
                message: str, progress: float, phase: str, phase_progress: float
            ) -> None:
                """Forward progress updates to WebSocket"""
                await ws_manager.send_progress(job_id, message, progress, phase, phase_progress)

            # Run JSON-based pipeline (detection → classification → database)
            result = await pipeline.process_deployment(
                deployment_id=deployment.id,
                deployment_folder=folder_path,  # Use folder_path from payload, not deployment record
                image_paths=image_files,
                job_id=job_id,
                db=db,
                progress_callback=progress_callback,
            )

            # Update job status
            job_crud.update_job_status(db, job_id, "completed")

            # Update queue entry if this job was from queue
            if queue_entry_id:
                queue_crud.update_queue_status(
                    db,
                    queue_entry_id,
                    status="completed",
                    deployment_id=deployment.id
                )
                logger.info(f"Updated queue entry {queue_entry_id} to completed")

            # Prepare completion message
            completion_message = f"Analysis complete: {result.total_detections} detections"
            if classification_model:
                completion_message += f", {result.classified_detections} classified"

            # Send completion message
            await ws_manager.send_complete(
                job_id=job_id,
                success=True,
                message=completion_message,
                data={
                    "deployment_id": deployment.id,
                    "file_count": result.total_files,
                    "detection_count": result.total_detections,
                    "animal_count": result.animal_detections,
                    "person_count": result.person_detections,
                    "vehicle_count": result.vehicle_detections,
                    "classified_count": result.classified_detections,
                },
            )

            logger.info(
                f"Job {job_id} completed: {result.total_files} files, "
                f"{result.total_detections} detections, "
                f"{result.classified_detections} classified"
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
                        db,
                        queue_entry_id,
                        status="failed",
                        error=str(e)
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

    for root, _, files in os.walk(folder_path):
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
    country_code: str | None,
    state_code: str | None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> None:
    """
    Run classification on detection JSON file.

    Handles both SpeciesNet (batch) and regular (per-detection) classifiers.
    Updates JSON file in-place with classification results.

    Args:
        json_path: Path to detection JSON file
        classification_model: Classification model instance
        deployment_folder: Deployment folder for artifacts
        country_code: Country code for SpeciesNet
        state_code: State code for SpeciesNet
        progress_callback: Optional progress callback

    Raises:
        RuntimeError: If classification fails
    """
    import json
    from app.ml.json_utils import extract_animal_detections

    # Check if SpeciesNet (batch processing)
    is_speciesnet = hasattr(classification_model, "classify_batch")

    if is_speciesnet:
        # SpeciesNet: Batch processing
        logger.info("Running SpeciesNet batch classification")

        # SpeciesNet uses old 4-argument progress callback format
        # Create adapter function to convert to new 2-argument format
        async def speciesnet_progress_adapter(message, progress, phase, phase_progress):
            """Adapter for SpeciesNet's old 4-argument progress callback format"""
            if progress_callback:
                await progress_callback(message, phase_progress)

        await classification_model.classify_batch(
            detection_json_path=json_path,
            country_code=country_code,
            state_code=state_code,
            deployment_folder=deployment_folder,
            progress_callback=speciesnet_progress_adapter,
        )
    else:
        # Regular per-detection classification
        logger.info("Running per-detection classification")

        # Load detection JSON
        with open(json_path, "r") as f:
            md_results = json.load(f)

        # Extract animal detections
        animal_detections = extract_animal_detections(md_results)
        total_animals = len(animal_detections)

        if total_animals == 0:
            logger.info("No animals to classify")
            return

        # Start classification worker (context manager)
        with classification_model as cls_model:
            # Get class names (ID -> name mapping)
            class_names = cls_model.get_class_names()

            # Create reverse mapping (name -> ID) for JSON creation
            name_to_id = {name: class_id for class_id, name in class_names.items()}

            # Group detections by file for efficient video frame caching
            from collections import defaultdict
            detections_by_file = defaultdict(list)
            for img_idx, det_idx, detection in animal_detections:
                img_info = md_results["images"][img_idx]
                relative_file = img_info["file"]
                img_path = (deployment_folder / relative_file).resolve()

                detections_by_file[str(img_path)].append({
                    'img_idx': img_idx,
                    'det_idx': det_idx,
                    'detection': detection,
                    'img_info': img_info
                })

            # Process files in order (videos first, then images)
            classified_count = 0
            processed_count = 0
            frame_cache = {}  # Cache for video frames

            for file_path_str, file_detections in detections_by_file.items():
                file_path = Path(file_path_str)

                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}, skipping {len(file_detections)} detections")
                    processed_count += len(file_detections)
                    continue

                is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS

                # For videos: extract all needed frames to cache first
                if is_video:
                    logger.debug(f"Extracting frames for video: {file_path.name}")

                    # Get unique frame numbers for this video
                    frame_numbers = set()
                    for det_info in file_detections:
                        frame_num = det_info['detection'].get('frame_number')
                        if frame_num is not None:
                            frame_numbers.add(frame_num)

                    if not frame_numbers:
                        logger.warning(f"No frame numbers found for video {file_path.name}, skipping")
                        processed_count += len(file_detections)
                        continue

                    # Extract frames to cache
                    from app.utils.video_utils import run_callback_on_frames, _frame_number_to_filename
                    from PIL import Image

                    def frame_callback(image_np, frame_id):
                        """Store frame as PIL Image in cache"""
                        frame_cache[frame_id] = Image.fromarray(image_np)
                        return None

                    try:
                        run_callback_on_frames(
                            str(file_path),
                            frame_callback,
                            frames_to_process=list(frame_numbers),
                            verbose=False
                        )
                        logger.debug(f"Extracted {len(frame_cache)} frames for {file_path.name}")
                    except Exception as e:
                        logger.error(f"Failed to extract frames from {file_path.name}: {e}")
                        processed_count += len(file_detections)
                        frame_cache.clear()
                        continue

                # Classify all detections in this file
                for det_info in file_detections:
                    try:
                        processed_count += 1

                        # Update progress
                        if progress_callback:
                            phase_progress = processed_count / total_animals
                            await progress_callback(
                                f"Classifying {processed_count}/{total_animals} animals",
                                phase_progress,
                            )

                        detection = det_info['detection']
                        img_idx = det_info['img_idx']
                        det_idx = det_info['det_idx']

                        # Get frame/image for classification
                        if is_video:
                            # Get frame from cache
                            frame_number = detection.get('frame_number')
                            if frame_number is None:
                                logger.warning(f"Detection missing frame_number, skipping")
                                continue

                            frame_key = _frame_number_to_filename(frame_number)
                            frame_image = frame_cache.get(frame_key)

                            if frame_image is None:
                                logger.warning(f"Frame {frame_key} not in cache, skipping")
                                continue
                        else:
                            # Load image from disk
                            from PIL import Image
                            frame_image = Image.open(file_path)

                        # Create bbox object
                        from app.ml.inference.base import BoundingBox
                        bbox = BoundingBox(
                            x=detection["bbox"][0],
                            y=detection["bbox"][1],
                            width=detection["bbox"][2],
                            height=detection["bbox"][3],
                        )

                        # For videos: save full frame to temp file (worker needs a file path and will crop it)
                        # For images: pass original file path directly (worker will load and crop it)
                        if is_video:
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                                tmp_path = Path(tmp.name)
                                frame_image.save(tmp_path)  # Save FULL FRAME, worker will crop

                            try:
                                result = cls_model.classify(tmp_path, bbox)
                            finally:
                                tmp_path.unlink(missing_ok=True)
                        else:
                            # For images: pass original file path (no temp file needed)
                            # Worker will load the full image and crop it using the bbox
                            result = cls_model.classify(file_path, bbox)

                        # Check if classification succeeded
                        if result is None:
                            logger.warning(f"Classification returned None for detection, skipping")
                            continue

                        # Add classification to detection (update in-place in md_results)
                        # Convert class names to IDs for JSON format consistency
                        md_results["images"][img_idx]["detections"][det_idx]["classifications"] = [
                            [name_to_id[class_name], prob]
                            for class_name, prob in result.all_probabilities.items()
                            if class_name in name_to_id
                        ][:10]  # Top 10 results

                        classified_count += 1

                    except Exception as e:
                        logger.error(f"Classification failed for detection: {e}")
                        continue

                # Clear video frame cache after processing this video
                if is_video:
                    frame_cache.clear()
                    logger.debug(f"Cleared frame cache for {file_path.name}")

            # Update class names in JSON
            if class_names:
                md_results["classification_categories"] = class_names

        # Save updated JSON
        with open(json_path, "w") as f:
            json.dump(md_results, f, indent=2)

        logger.info(f"Classified {classified_count}/{total_animals} animals")


def merge_json_files(json_files: list[Path], output_file: Path, deployment_id: str) -> None:
    """
    Merge multiple JSON files (video and image results) into single file.

    IMPORTANT: This function properly handles SpeciesNet's dynamic classification IDs
    by creating a unified classification_categories mapping and renumbering all
    classification IDs to be consistent across video and image detections.

    Why this is necessary:
    - SpeciesNet assigns classification IDs dynamically based on the order species appear
    - Video and image JSONs may have different ID mappings for the same species
    - Example: "zebra" might be ID "1" in video JSON but ID "2" in image JSON
    - This function unifies the mappings so all IDs are consistent

    Args:
        json_files: List of JSON file paths to merge
        output_file: Output merged JSON file path
        deployment_id: Deployment ID for metadata

    Raises:
        RuntimeError: If merge fails
    """
    import json

    try:
        merged_data = {
            "images": [],
            "detection_categories": {},
            "classification_categories": {},
            "info": {},
        }

        # Track unified classification mapping: species_name -> unified_id
        unified_class_mapping = {}
        next_class_id = 1

        for json_file in json_files:
            if not json_file.exists():
                logger.warning(f"JSON file not found: {json_file}")
                continue

            with open(json_file, "r") as f:
                data = json.load(f)

            # Get classification categories from this file
            file_class_categories = data.get("classification_categories", {})

            # Build mapping from old ID to new ID for this file
            id_remapping = {}

            # For each species in this file's classification_categories
            for old_id, species_name in file_class_categories.items():
                # Check if we've seen this species before
                if species_name not in unified_class_mapping:
                    # New species - assign next available ID
                    unified_class_mapping[species_name] = str(next_class_id)
                    next_class_id += 1

                # Map old ID to unified ID
                id_remapping[old_id] = unified_class_mapping[species_name]

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
                                        f"Unknown classification ID '{class_id}' in {json_file.name}, "
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
            class_id: species_name
            for species_name, class_id in unified_class_mapping.items()
        }

        logger.info(
            f"Unified classification mapping: {len(merged_data['classification_categories'])} species "
            f"across {len(json_files)} JSON files"
        )

        # Add metadata
        merged_data["addaxai_metadata"] = {
            "deployment_id": deployment_id,
            "processed_at": datetime.utcnow().isoformat(),
        }

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

    for root, _, files in os.walk(folder_path):
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
