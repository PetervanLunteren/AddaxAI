"""
Detection worker for deployment analysis jobs.

Following DEVELOPERS.md principles:
- Crash early if configuration invalid
- Explicit error handling
- Type hints everywhere
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

from app.api.crud import detection as detection_crud
from app.api.crud import deployment as deployment_crud
from app.api.crud import job as job_crud
from app.api.crud import site as site_crud
from app.api.schemas.detection import DetectionCreate
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.session import get_db
from app.ml.detection import MegaDetectorRunner
from app.ml.environment_manager import EnvironmentManager
from app.ml.manifest_manager import ManifestManager
from app.models import Deployment, File

logger = get_logger(__name__)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


async def process_deployment_analysis(job_id: str) -> None:
    """
    Process deployment analysis job (detection phase).

    Workflow:
    1. Get job and validate payload
    2. Scan folder for images
    3. Create deployment record
    4. Run MegaDetector on images
    5. Create File and Detection records
    6. Update job status

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
            folder_path = payload.get("folder_path")
            detection_model = payload.get("detection_model")

            if not all([project_id, folder_path, detection_model]):
                raise ValueError("Invalid job payload: missing required fields")

            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise ValueError(f"Folder not found: {folder_path}")

            # Update job status
            job_crud.update_job_status(db, job_id, "running")

            # Scan folder for images
            await ws_manager.send_progress(job_id, "Scanning folder for images...", 0.05)
            image_files = scan_folder_for_images(folder_path)
            logger.info(f"Found {len(image_files)} images in {folder_path}")

            if not image_files:
                raise ValueError(f"No images found in {folder_path}")

            # Get or create "Unknown Site"
            await ws_manager.send_progress(job_id, "Creating deployment...", 0.10)
            site = site_crud.get_or_create_unknown_site(db, project_id)

            # Create deployment
            deployment = create_deployment(
                db=db,
                site_id=site.id,
                folder_path=str(folder_path),
            )
            logger.info(f"Created deployment: {deployment.id}")

            # Run MegaDetector
            await ws_manager.send_progress(job_id, "Running MegaDetector...", 0.15)

            # Setup ML infrastructure
            manifest_manager = ManifestManager()
            env_manager = EnvironmentManager()
            runner = MegaDetectorRunner(env_manager)

            # Get model manifest
            manifest = manifest_manager.get_model(detection_model)

            # Progress callback for MegaDetector
            def progress_callback(message: str, progress: float) -> None:
                """Send progress update (sync wrapper for async)."""
                asyncio.create_task(ws_manager.send_progress(job_id, message, progress))

            # Run detection
            results = runner.run_detection(
                manifest=manifest,
                image_paths=image_files,
                confidence_threshold=0.1,
                progress_callback=progress_callback,
            )

            # Process results and create File + Detection records
            await ws_manager.send_progress(job_id, "Saving results to database...", 0.95)
            file_count, detection_count = save_detection_results(
                db=db,
                deployment_id=deployment.id,
                job_id=job_id,
                folder_path=folder_path,
                results=results,
            )

            # Update job status
            job_crud.update_job_status(db, job_id, "completed")

            # Send completion message
            await ws_manager.send_complete(
                job_id=job_id,
                success=True,
                message="Detection complete",
                data={
                    "deployment_id": deployment.id,
                    "file_count": file_count,
                    "detection_count": detection_count,
                },
            )

            logger.info(
                f"Job {job_id} completed: {file_count} files, {detection_count} detections"
            )

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        # Update job status
        try:
            db = next(get_db())
            job_crud.update_job_status(db, job_id, "failed")
            db.close()
        except Exception:
            pass

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


def save_detection_results(
    db,
    deployment_id: str,
    job_id: str,
    folder_path: Path,
    results: dict,
) -> tuple[int, int]:
    """
    Save detection results to database.

    Creates File and Detection records for each image.

    Args:
        db: Database session
        deployment_id: Deployment ID
        job_id: Job ID
        folder_path: Deployment folder path
        results: MegaDetector results dict

    Returns:
        Tuple of (file_count, detection_count)
    """
    file_count = 0
    detection_count = 0

    # Category mapping: MegaDetector uses "1"=animal, "2"=person, "3"=vehicle
    CATEGORY_MAP = {
        "1": "animal",
        "2": "person",
        "3": "vehicle",
    }

    for image_result in results.get("images", []):
        image_path = Path(image_result["file"])

        # Extract EXIF timestamp
        timestamp = extract_timestamp_from_exif(image_path)

        # Create File record
        file_record = File(
            deployment_id=deployment_id,
            file_path=str(image_path),
            file_type="image",
            file_format=image_path.suffix.lstrip(".").lower(),
            size_bytes=image_path.stat().st_size if image_path.exists() else None,
            timestamp=timestamp,
        )

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                file_record.width_px = img.width
                file_record.height_px = img.height
        except Exception as e:
            logger.warning(f"Failed to read image dimensions for {image_path}: {e}")

        db.add(file_record)
        db.flush()  # Get file_record.id
        file_count += 1

        # Create Detection records
        detections_data: list[DetectionCreate] = []

        for det in image_result.get("detections", []):
            category_num = str(det["category"])
            category = CATEGORY_MAP.get(category_num, "animal")

            bbox = det["bbox"]  # [x, y, width, height]

            detection_data = DetectionCreate(
                file_id=file_record.id,
                job_id=job_id,
                category=category,
                confidence=det["conf"],
                bbox_x=bbox[0],
                bbox_y=bbox[1],
                bbox_width=bbox[2],
                bbox_height=bbox[3],
            )
            detections_data.append(detection_data)

        # Bulk create detections
        if detections_data:
            detection_crud.create_detections_bulk(db, detections_data)
            detection_count += len(detections_data)

    db.commit()

    return file_count, detection_count


def extract_timestamp_from_exif(image_path: Path) -> datetime:
    """
    Extract timestamp from EXIF data.

    Falls back to file modification time if EXIF not available.

    Args:
        image_path: Path to image file

    Returns:
        Timestamp as datetime
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()

            if exif_data:
                # Look for DateTimeOriginal tag (0x9003)
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal":
                        # Parse EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")

    except Exception as e:
        logger.debug(f"Failed to read EXIF from {image_path}: {e}")

    # Fallback to file modification time
    return datetime.fromtimestamp(image_path.stat().st_mtime)
