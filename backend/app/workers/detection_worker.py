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

# Supported video formats (from MegaDetector video_utils)
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

            # Scan folder for images and videos
            image_files = scan_folder_for_images(folder_path)
            video_files = scan_folder_for_videos(folder_path)
            logger.info(
                f"Found {len(image_files)} images and {len(video_files)} videos in {folder_path}"
            )

            if not image_files and not video_files:
                logger.warning(f"No images or videos found in {folder_path}, skipping")
                queue_crud.update_queue_status(db, entry_id, status="completed")
                continue

            # Extract frames from videos if present
            temp_frames_folder = None
            video_frame_mapping = {}  # Maps video_path -> (frame_paths, frame_rate)

            if video_files:
                temp_frames_folder = folder_path / ".addaxai" / "temp_frames"
                temp_frames_folder.mkdir(parents=True, exist_ok=True)

                await ws_manager.send_progress(
                    job_id,
                    f"[{idx}/{total_entries}] Extracting frames from {len(video_files)} videos...",
                    progress_start + 0.05 * progress_range
                )

                for video_path in video_files:
                    try:
                        frame_paths, frame_rate = extract_video_frames(
                            video_path, project.video_fps, temp_frames_folder
                        )
                        video_frame_mapping[video_path] = (frame_paths, frame_rate)
                        image_files.extend(frame_paths)  # Add frames to processing list
                    except Exception as e:
                        logger.error(f"Failed to process video {video_path}: {e}")
                        # Continue with other videos

                logger.info(
                    f"Extracted {sum(len(paths) for paths, _ in video_frame_mapping.values())} "
                    f"total frames from videos"
                )

            total_files += len(image_files)

            # Get or create "Unknown Site"
            site = site_crud.get_or_create_unknown_site(db, project_id)

            # Create deployment
            deployment = create_deployment(db=db, site_id=site.id, folder_path=str(folder_path))
            logger.info(f"Created deployment: {deployment.id}")

            # Define progress callback for this specific deployment
            async def deployment_progress_callback(
                message: str, progress: float, phase: str, phase_progress: float
            ) -> None:
                """Forward progress updates with deployment number prefix"""
                # Scale progress to this deployment's range within the overall batch
                overall_progress = progress_start + (progress * progress_range)
                await ws_manager.send_progress(
                    job_id,
                    f"[{idx}/{total_entries}] {message}",
                    overall_progress,
                    phase,
                    phase_progress
                )

            # Run JSON-based pipeline for this deployment
            result = await pipeline.process_deployment(
                deployment_id=deployment.id,
                deployment_folder=folder_path,
                image_paths=image_files,
                job_id=job_id,
                db=db,
                progress_callback=deployment_progress_callback,
            )

            total_detections += result.total_detections

            # Clean up temporary video frames
            if temp_frames_folder and temp_frames_folder.exists():
                import shutil
                try:
                    shutil.rmtree(temp_frames_folder)
                    logger.info(f"Cleaned up temporary frames from {temp_frames_folder}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp frames: {e}")

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


def extract_video_frames(video_path: Path, fps: float, temp_folder: Path) -> tuple[list[Path], float]:
    """
    Extract frames from video at specified FPS.

    Uses video_utils from streamlit-AddaxAI (proven approach).

    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        temp_folder: Temporary folder for frame extraction

    Returns:
        Tuple of (list of frame paths, actual video frame rate)

    Raises:
        RuntimeError: If frame extraction fails
    """
    from app.utils.video_utils import video_to_frames

    # Create unique subfolder for this video's frames
    video_frames_folder = temp_folder / video_path.stem
    video_frames_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Extract frames using negative fps value (sampling rate in seconds)
        # fps=2.0 means extract every 0.5 seconds → every_n_frames=-0.5
        every_n_seconds = 1.0 / fps

        logger.info(f"Extracting frames from {video_path.name} at {fps} FPS")

        frame_filenames, video_frame_rate = video_to_frames(
            input_video_file=str(video_path),
            output_folder=str(video_frames_folder),
            overwrite=True,
            every_n_frames=-every_n_seconds,  # Negative = time-based sampling
            verbose=False,
            quality=None,  # Use OpenCV default
            allow_empty_videos=False,
        )

        # Convert to Path objects
        frame_paths = [Path(f) for f in frame_filenames]

        logger.info(
            f"Extracted {len(frame_paths)} frames from {video_path.name} "
            f"(video FPS: {video_frame_rate})"
        )

        return frame_paths, video_frame_rate

    except Exception as e:
        logger.error(f"Failed to extract frames from {video_path}: {e}", exc_info=True)
        raise RuntimeError(f"Frame extraction failed for {video_path.name}: {e}") from e


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
