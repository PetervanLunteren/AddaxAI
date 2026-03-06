"""
ML Pipeline Orchestrator

Coordinates detection → classification pipeline with database integration.

Following DEVELOPERS.MD principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere
- Clean async/await integration

Created by Claude Code on 2026-01-04
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.api.crud import detection as detection_crud
from app.api.schemas.detection import DetectionCreate
from app.core.logging_config import get_logger
from app.ml.inference.base import (
    BoundingBox,
    ClassificationModel,
    DetectionModel,
    PipelineResult,
)
from app.models import Detection, File

logger = get_logger(__name__)


class MLPipeline:
    """
    Orchestrates detection → classification pipeline.

    Clean, async-compatible, database-integrated pipeline that:
    1. Runs detection model on images
    2. Saves detections to database
    3. Runs classification on animal detections
    4. Updates database with species information
    5. Reports progress via WebSocket callback
    """

    def __init__(
        self,
        detection_model: DetectionModel,
        classification_model: ClassificationModel | None,
    ):
        """
        Initialize ML pipeline.

        Args:
            detection_model: Detection model (e.g., MegaDetectorV1000)
            classification_model: Optional classification model (e.g., YOLOv8Classifier)
        """
        self.detection_model = detection_model
        self.classification_model = classification_model

        cls_name = (
            type(classification_model).__name__
            if classification_model
            else "None"
        )
        logger.info(
            f"Pipeline initialized with "
            f"detection={type(detection_model).__name__}, "
            f"classification={cls_name}"
        )

    async def process_deployment(
        self,
        deployment_id: str,
        image_paths: list[Path],
        job_id: str,
        db: Session,
        progress_callback: Callable[[str, float, str, float], None],
    ) -> PipelineResult:
        """
        Run complete ML pipeline on a deployment.

        Workflow:
        1. Run detection model → DetectionResult objects
        2. Save to database → File + Detection records
        3. Run classification (if model configured) → ClassificationResult objects
        4. Update database → Detection.species + Detection.species_confidence

        Args:
            deployment_id: Deployment ID for database records
            image_paths: List of absolute paths to images
            job_id: Job ID for tracking
            db: Database session
            progress_callback: Async callback(message, overall_progress, phase, phase_progress)

        Returns:
            PipelineResult with statistics

        Raises:
            RuntimeError: If pipeline fails
        """
        try:
            logger.info(
                f"Starting pipeline for deployment {deployment_id} "
                f"with {len(image_paths)} images"
            )

            # Phase 0: Initialization
            await progress_callback("Initializing ML models...", 0.0, "init", 0.0)

            # Phase 1: Detection
            await progress_callback("Detection: Starting...", 0.1, "detection", 0.0)

            # Detection runs synchronously and blocks the event loop
            # Run it in a thread pool so the event loop can continue processing WebSocket messages
            loop = asyncio.get_event_loop()

            # Create thread-safe progress callback
            # MegaDetector runs in thread and needs to send progress to async event loop
            def sync_progress_callback(message: str, progress: float) -> None:
                """Thread-safe wrapper that schedules async callback on event loop"""
                # Add "Detection: " prefix to message
                prefixed_message = f"Detection: {message}"
                # Overall progress: 0.1 to 0.5 for detection phase
                overall_progress = 0.1 + (progress * 0.4)
                # Schedule callback on event loop from thread
                asyncio.run_coroutine_threadsafe(
                    progress_callback(prefixed_message, overall_progress, "detection", progress),
                    loop,
                )

            detections = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                lambda: self.detection_model.detect(
                    image_paths=image_paths,
                    confidence_threshold=0.1,
                    progress_callback=sync_progress_callback,
                ),
            )

            logger.info(f"Detection found {len(detections)} total detections")

            await progress_callback("Detection: Complete", 0.5, "detection", 1.0)

            # Phase 2: Finalization - Save detections to database
            await progress_callback("Saving detections to database...", 0.51, "finalize", 0.1)

            detection_records = self._save_detections_to_db(
                db=db,
                deployment_id=deployment_id,
                job_id=job_id,
                detections=detections,
                all_image_paths=image_paths,
            )

            logger.info(f"Saved {len(detection_records)} detection records to database")

            # Count by category
            animal_count = sum(1 for d in detection_records if d.category == "animal")
            person_count = sum(1 for d in detection_records if d.category == "person")
            vehicle_count = sum(1 for d in detection_records if d.category == "vehicle")

            logger.info(
                f"Detections by category: animal={animal_count}, "
                f"person={person_count}, vehicle={vehicle_count}"
            )

            # Phase 3: Classification (if model configured)
            classified_count = 0
            if self.classification_model and animal_count > 0:
                await progress_callback(
                    f"Classification: Starting on {animal_count} animals...",
                    0.55,
                    "classification",
                    0.0,
                )

                # Filter animal detections
                animal_detections = [d for d in detection_records if d.category == "animal"]

                # Use classification model as context manager (starts/stops worker)
                with self.classification_model:
                    for i, detection in enumerate(animal_detections):
                        try:
                            # Load image
                            image = Image.open(detection.file.file_path)

                            # Create BoundingBox
                            bbox = BoundingBox(
                                x=detection.bbox_x,
                                y=detection.bbox_y,
                                width=detection.bbox_width,
                                height=detection.bbox_height,
                            )

                            # Run classification
                            result = self.classification_model.classify(
                                image=image,
                                bbox=bbox,
                                progress_callback=None,
                            )

                            # Handle skipped detections (result is None)
                            if result is not None:
                                # Update detection record
                                detection.species = result.species
                                detection.species_confidence = result.confidence
                                detection.classification_all_probs = result.all_probabilities
                                detection.classification_method = "machine"
                                classified_count += 1

                            # Update progress after EVERY detection (smooth progress bar)
                            phase_progress = (i + 1) / len(animal_detections)
                            overall_progress = 0.55 + (phase_progress * 0.35)
                            n_animals = len(animal_detections)
                            await progress_callback(
                                f"Classification: {classified_count}"
                                f"/{n_animals} animals"
                                f" ({i + 1} processed)",
                                overall_progress,
                                "classification",
                                phase_progress,
                            )

                        except Exception as e:
                            logger.error(
                                f"Classification failed for detection {detection.id}: {e}",
                                exc_info=True,
                            )
                            # Continue with next detection instead of failing entire pipeline

                # Commit all classification updates
                db.commit()
                logger.info(f"Successfully classified {classified_count} animals")
            elif animal_count == 0:
                # No animals detected, skip classification
                await progress_callback(
                    "No animals detected - skipping classification", 0.55, "classification", 0.0
                )

            # Phase 4: Final finalization
            await progress_callback("Updating deployment status...", 0.95, "finalize", 0.8)

            await progress_callback("Pipeline complete", 1.0, "finalize", 1.0)

            result = PipelineResult(
                total_files=len(set(d.file_path for d in detections)),
                total_detections=len(detections),
                animal_detections=animal_count,
                person_detections=person_count,
                vehicle_detections=vehicle_count,
                classified_detections=classified_count,
            )

            logger.info(
                f"Pipeline completed: {result.total_files} files, "
                f"{result.total_detections} detections, "
                f"{result.classified_detections} classified"
            )

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"ML pipeline execution failed: {e}") from e

    def _save_detections_to_db(
        self,
        db: Session,
        deployment_id: str,
        job_id: str,
        detections: list,
        all_image_paths: list[Path] | None = None,
    ) -> list[Detection]:
        """
        Save detection results to database.

        Creates File records (if they don't exist) and Detection records.
        Also creates blank File records for images with no detections.

        Args:
            db: Database session
            deployment_id: Deployment ID
            job_id: Job ID
            detections: List of DetectionResult objects
            all_image_paths: All scanned image paths (for creating blank File records)

        Returns:
            List of created Detection database records (with relationships loaded)
        """
        # Group detections by file
        from collections import defaultdict
        from datetime import datetime

        file_detections = defaultdict(list)
        for det in detections:
            file_detections[det.file_path].append(det)

        detection_records = []

        for file_path, file_dets in file_detections.items():
            # Check if File record exists
            file_record = db.query(File).filter(File.file_path == str(file_path)).first()

            if not file_record:
                # Create File record
                file_record = File(
                    deployment_id=deployment_id,
                    file_path=str(file_path),
                    file_type="image",
                    file_format=file_path.suffix.lstrip(".").lower(),
                    size_bytes=file_path.stat().st_size if file_path.exists() else None,
                    timestamp=datetime.fromtimestamp(file_path.stat().st_mtime),
                )

                # Get image dimensions
                try:
                    with Image.open(file_path) as img:
                        file_record.width_px = img.width
                        file_record.height_px = img.height
                except Exception as e:
                    logger.warning(f"Failed to read image dimensions for {file_path}: {e}")

                db.add(file_record)
                db.flush()  # Get file_record.id

            # Set observation_type based on detection categories
            # Priority: animal > human > vehicle
            categories = {det.category for det in file_dets}
            if "animal" in categories:
                file_record.observation_type = "animal"
            elif "person" in categories:
                file_record.observation_type = "human"
            elif "vehicle" in categories:
                file_record.observation_type = "vehicle"

            # Create Detection records for this file
            for det in file_dets:
                detection_data = DetectionCreate(
                    file_id=file_record.id,
                    job_id=job_id,
                    category=det.category,
                    confidence=det.confidence,
                    bbox_x=det.bbox.x,
                    bbox_y=det.bbox.y,
                    bbox_width=det.bbox.width,
                    bbox_height=det.bbox.height,
                )

                # Create detection record
                detection_record = detection_crud.create_detection(db, detection_data)

                # Manually load file relationship for classification phase
                detection_record.file = file_record

                detection_records.append(detection_record)

        # Create blank File records for images with no detections
        if all_image_paths:
            files_with_detections = {str(p) for p in file_detections.keys()}
            for image_path in all_image_paths:
                if str(image_path) not in files_with_detections:
                    # Check if File record already exists
                    existing = db.query(File).filter(File.file_path == str(image_path)).first()
                    if not existing:
                        blank_file = File(
                            deployment_id=deployment_id,
                            file_path=str(image_path),
                            file_type="image",
                            file_format=image_path.suffix.lstrip(".").lower(),
                            size_bytes=image_path.stat().st_size if image_path.exists() else None,
                            timestamp=datetime.fromtimestamp(image_path.stat().st_mtime),
                            observation_type="blank",
                        )
                        try:
                            with Image.open(image_path) as img:
                                blank_file.width_px = img.width
                                blank_file.height_px = img.height
                        except Exception as e:
                            logger.warning(f"Failed to read image dimensions for {image_path}: {e}")
                        db.add(blank_file)

        db.commit()

        return detection_records
