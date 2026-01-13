"""
JSON-Based ML Pipeline Orchestrator

Sequential pipeline: MegaDetector → JSON → Classification → Extended JSON → DB
Battle-tested approach matching streamlit-AddaxAI architecture.

Following DEVELOPERS.MD principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere
- Clean async/await integration

Created by Claude Code on 2026-01-05
"""

import asyncio
import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from app.api.crud import detection as detection_crud
from app.api.schemas.detection import DetectionCreate
from app.core.logging_config import get_logger
from app.ml.inference.base import ClassificationModel, DetectionModel, PipelineResult
from app.ml.json_utils import (
    assign_uuids_to_detection_json,
    build_addaxai_metadata,
    extract_animal_detections,
)
from app.models import Detection, File

logger = get_logger(__name__)


class JSONBasedMLPipeline:
    """
    JSON-based ML pipeline with dual progress bars.

    Architecture:
    1. MegaDetector → detection_results.json (pure MD format)
    2. Load JSON → Classify detections → results_with_classifications.json (extended)
    3. Parse extended JSON → Bulk insert to database
    4. Save JSON artifacts in .addaxai folder

    Progress Bars (Two Separate):
    - Detection: 0-100% during MegaDetector phase
    - Classification: 0-100% during classification phase
    - Init/Finalize: Shown as spinner phases
    """

    def __init__(
        self,
        detection_model: DetectionModel,
        classification_model: ClassificationModel | None,
        detection_model_id: str,
        classification_model_id: str | None,
        country_code: str | None = None,
        state_code: str | None = None,
    ):
        """
        Initialize JSON-based ML pipeline.

        Args:
            detection_model: Detection model (e.g., MegaDetectorV1000)
            classification_model: Optional classification model
            detection_model_id: Detection model ID for metadata
            classification_model_id: Classification model ID for metadata
            country_code: Country code for SpeciesNet geofencing (e.g., "USA", "KEN")
            state_code: State code for USA (e.g., "CA", "TX")
        """
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.detection_model_id = detection_model_id
        self.classification_model_id = classification_model_id
        self.country_code = country_code
        self.state_code = state_code

        logger.info(
            f"JSON Pipeline initialized with detection={type(detection_model).__name__}, "
            f"classification={type(classification_model).__name__ if classification_model else 'None'}"
        )

    async def process_deployment(
        self,
        deployment_id: str,
        deployment_folder: Path,
        image_paths: list[Path],
        job_id: str,
        db: Session,
        progress_callback: Callable[[str, float, str, float], None],
    ) -> PipelineResult:
        """
        Run complete JSON-based ML pipeline on a deployment.

        Workflow:
        1. Initialize progress bars (both at 0%)
        2. Run MegaDetector → save detection_results.json
        3. Load JSON → Classify animals → save results_with_classifications.json
        4. Parse extended JSON → Bulk insert to database
        5. Return statistics

        Args:
            deployment_id: Deployment ID for database records
            deployment_folder: Path to deployment folder (for .addaxai artifacts)
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
                f"Starting JSON pipeline for deployment {deployment_id} "
                f"with {len(image_paths)} images"
            )
            logger.info(f"Deployment folder: {deployment_folder}")
            logger.info(f"Deployment folder exists: {deployment_folder.exists()}")

            # Phase 0: Initialization (show both progress bars at 0%)
            await progress_callback("Initializing ML models...", 0.0, "init", 0.0)
            await progress_callback("Detection: Waiting...", 0.0, "detection", 0.0)
            await progress_callback(
                "Classification: Waiting for detection...", 0.0, "classification", 0.0
            )
            await asyncio.sleep(0.1)

            # Phase 1: Detection (updates detection progress bar)
            logger.info("Phase 1: Running MegaDetector")
            detection_json_path = await self._run_detection(
                image_paths=image_paths,
                deployment_folder=deployment_folder,
                progress_callback=progress_callback,
            )

            # Detection complete - update progress bars
            await progress_callback("Detection complete", 0.4, "detection", 1.0)
            logger.info(f"Detection JSON saved to: {detection_json_path}")

            # Phase 2: Classification (updates classification progress bar)
            classified_count = 0
            if self.classification_model:
                logger.info("Phase 2: Running classification")
                extended_json_path, classified_count = await self._run_classification(
                    detection_json_path=detection_json_path,
                    deployment_id=deployment_id,
                    deployment_folder=deployment_folder,
                    progress_callback=progress_callback,
                )

                await progress_callback("Classification complete", 0.8, "classification", 1.0)
                logger.info(
                    f"Extended JSON saved to: {extended_json_path}, classified {classified_count} animals"
                )
            else:
                # No classification model - use detection JSON as final JSON
                extended_json_path = detection_json_path
                logger.info("Phase 2: Skipping classification (no model configured)")

            # Phase 3: Database Load (finalize phase)
            await progress_callback("Loading results to database...", 0.9, "finalize", 0.0)
            logger.info("Phase 3: Loading to database")

            result = self._load_to_database(
                extended_json_path=extended_json_path,
                deployment_id=deployment_id,
                deployment_folder=deployment_folder,
                job_id=job_id,
                db=db,
            )

            await progress_callback("Pipeline complete", 1.0, "finalize", 1.0)

            logger.info(
                f"JSON Pipeline completed: {result.total_files} files, "
                f"{result.total_detections} detections, "
                f"{result.classified_detections} classified"
            )

            return result

        except Exception as e:
            logger.error(f"JSON Pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"JSON-based ML pipeline execution failed: {e}") from e

    async def _run_detection(
        self,
        image_paths: list[Path],
        deployment_folder: Path,
        progress_callback: Callable,
    ) -> Path:
        """
        Run MegaDetector and save results to JSON.

        Updates detection progress bar (phase="detection", phase_progress=0-1).

        Args:
            image_paths: List of image paths
            deployment_folder: Deployment folder for artifacts
            progress_callback: Progress callback

        Returns:
            Path to detection_results.json
        """
        # Run detection in thread pool (MegaDetector is blocking)
        loop = asyncio.get_event_loop()

        # Create thread-safe progress callback for detection phase
        def detection_progress(message: str, progress: float) -> None:
            """Thread-safe wrapper for detection progress updates"""
            # Schedule callback on event loop from thread
            asyncio.run_coroutine_threadsafe(
                progress_callback(
                    f"Detection: {message}",
                    0.0,  # Overall progress (not used for dual progress bars)
                    "detection",
                    progress,
                ),
                loop,
            )

        # Run MegaDetector in thread pool
        detection_json_path = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            lambda: self.detection_model.detect_to_json(
                image_paths=image_paths,
                deployment_folder=deployment_folder,
                confidence_threshold=0.1,
                progress_callback=detection_progress,
            ),
        )

        return detection_json_path

    async def _run_classification(
        self,
        detection_json_path: Path,
        deployment_id: str,
        deployment_folder: Path,
        progress_callback: Callable,
    ) -> tuple[Path, int]:
        """
        Load detection JSON, classify animals, save extended JSON.

        Updates classification progress bar (phase="classification", phase_progress=0-1).

        Args:
            detection_json_path: Path to detection_results.json
            deployment_id: Deployment ID
            deployment_folder: Deployment folder for artifacts
            progress_callback: Progress callback

        Returns:
            Tuple of (extended_json_path, classified_count)
        """
        # Load detection JSON
        logger.info(f"Loading detection JSON from: {detection_json_path}")
        with open(detection_json_path) as f:
            md_results = json.load(f)

        # Extract animal detections
        animal_detections = extract_animal_detections(md_results)
        total_animals = len(animal_detections)

        logger.info(f"Found {total_animals} animal detections to classify")

        if total_animals == 0:
            await progress_callback(
                "Classification: No animals detected",
                0.0,
                "classification",
                0.0,
            )
            # Return detection JSON as-is (no classification needed)
            return detection_json_path, 0

        # Notify user of animal count
        await progress_callback(
            f"Classification: Starting on {total_animals} animals...",
            0.0,
            "classification",
            0.0,
        )

        # Branch: SpeciesNet (batch) vs Regular (per-detection)
        is_speciesnet = "SPECIESNET" in (self.classification_model_id or "").upper()

        if is_speciesnet:
            # SpeciesNet: Batch processing path
            logger.info("Using SpeciesNet batch processing")

            # Call classify_batch() - modifies detection JSON in-place
            await self.classification_model.classify_batch(
                detection_json_path=detection_json_path,
                country_code=self.country_code,
                state_code=self.state_code,
                deployment_folder=deployment_folder,
                progress_callback=progress_callback,
            )

            # Reload modified JSON
            with open(detection_json_path) as f:
                md_results = json.load(f)

            # Count classified animals
            classified_count = 0
            class_names = md_results.get("classification_categories", {})

            for img in md_results.get("images", []):
                for det in img.get("detections", []):
                    if "classifications" in det and det["classifications"]:
                        classified_count += 1

            logger.info(f"SpeciesNet classified {classified_count} animals")

            # Add addaxai_metadata
            md_results["addaxai_metadata"] = build_addaxai_metadata(
                deployment_id=deployment_id,
                det_model_id=self.detection_model_id,
                cls_model_id=self.classification_model_id,
                md_results=md_results,
            )

            # Save extended JSON
            artifacts_folder = deployment_folder / ".addaxai"
            artifacts_folder.mkdir(parents=True, exist_ok=True)
            extended_json_path = artifacts_folder / "results_with_classifications.json"

            with open(extended_json_path, "w") as f:
                json.dump(md_results, f, indent=2)

            return extended_json_path, classified_count

        # Regular per-detection classification path
        # Start classification worker
        classified_count = 0
        class_names = None

        with self.classification_model as cls_model:
            # Get class names from model
            class_names = cls_model.get_class_names()
            logger.info(f"Retrieved {len(class_names)} class names from model")

            # Assign UUIDs to detection JSON
            assign_uuids_to_detection_json(md_results)

            # Classify each animal detection
            for i, (img_idx, det_idx, det) in enumerate(animal_detections):
                try:
                    # Get image info
                    img_info = md_results["images"][img_idx]
                    relative_file = img_info["file"]

                    # Construct absolute path
                    image_path = (deployment_folder / relative_file).resolve()

                    if not image_path.exists():
                        logger.warning(f"Image not found: {image_path}, skipping detection")
                        continue

                    # Get bbox (normalized coordinates)
                    bbox = det["bbox"]  # [x, y, width, height]

                    # Classify detection (worker will load image from path)
                    result = cls_model.classify(
                        image_path=image_path,
                        bbox=BoundingBox(
                            x=float(bbox[0]),
                            y=float(bbox[1]),
                            width=float(bbox[2]),
                            height=float(bbox[3]),
                        ),
                        progress_callback=None,
                    )

                    # Handle skipped detections (result is None)
                    if result is not None:
                        # Convert classifications to expected format
                        # Result has all_probabilities as dict {species: confidence}
                        # We need list of [class_id, confidence] sorted by confidence

                        # Create inverse mapping: species_name -> class_id
                        species_to_id = {v: k for k, v in class_names.items()}

                        # Build classifications list
                        classifications = []
                        for species, conf in result.all_probabilities.items():
                            class_id = species_to_id.get(species)
                            if class_id is not None:
                                classifications.append([class_id, round(conf, 5)])

                        # Sort by confidence descending
                        classifications.sort(key=lambda x: x[1], reverse=True)

                        # Add classifications to detection
                        md_results["images"][img_idx]["detections"][det_idx][
                            "classifications"
                        ] = classifications

                        classified_count += 1

                    # Update progress after each detection
                    phase_progress = (i + 1) / total_animals
                    await progress_callback(
                        f"Classification: {classified_count}/{total_animals} animals (processed {i+1})",
                        0.0,
                        "classification",
                        phase_progress,
                    )

                except Exception as e:
                    logger.error(
                        f"Classification failed for detection {i+1}/{total_animals}: {e}",
                        exc_info=True,
                    )
                    # Continue with next detection

        # Add classification_categories to JSON
        md_results["classification_categories"] = class_names

        # Add addaxai_metadata
        md_results["addaxai_metadata"] = build_addaxai_metadata(
            deployment_id=deployment_id,
            det_model_id=self.detection_model_id,
            cls_model_id=self.classification_model_id,
            md_results=md_results,
        )

        # Save extended JSON
        artifacts_folder = deployment_folder / ".addaxai"
        artifacts_folder.mkdir(parents=True, exist_ok=True)
        extended_json_path = artifacts_folder / "results_with_classifications.json"

        with open(extended_json_path, "w") as f:
            json.dump(md_results, f, indent=2)

        logger.info(f"Successfully classified {classified_count}/{total_animals} animals")

        return extended_json_path, classified_count

    def _load_to_database(
        self,
        extended_json_path: Path,
        deployment_id: str,
        deployment_folder: Path,
        job_id: str,
        db: Session,
    ) -> PipelineResult:
        """
        Parse extended JSON and bulk insert to database.

        Creates File and Detection records from JSON data.

        Args:
            extended_json_path: Path to results_with_classifications.json
            deployment_id: Deployment ID
            deployment_folder: Deployment folder (base for relative paths)
            job_id: Job ID
            db: Database session

        Returns:
            PipelineResult with statistics
        """
        # Load extended JSON
        with open(extended_json_path) as f:
            results = json.load(f)

        logger.info(f"Loading {len(results.get('images', []))} images to database")

        # Track statistics
        total_detections = 0
        animal_count = 0
        person_count = 0
        vehicle_count = 0
        classified_count = 0

        # Group detections by file
        file_detections = defaultdict(list)

        for img in results.get("images", []):
            relative_file = img["file"]
            absolute_path = (deployment_folder / relative_file).resolve()

            # Get or create File record
            # First check by file_id if provided in JSON
            file_id = img.get("file_id")
            file_record = None

            if file_id:
                file_record = db.query(File).filter(File.id == file_id).first()

            # If not found by file_id, check by file_path AND deployment_id
            # This ensures each project gets its own file records even for the same physical file
            if not file_record:
                file_record = (
                    db.query(File)
                    .filter(File.file_path == str(absolute_path))
                    .filter(File.deployment_id == deployment_id)
                    .first()
                )

            # Create new file record if still not found
            if not file_record:
                if not file_id:
                    file_id = str(uuid.uuid4())

                file_record = File(
                    id=file_id,
                    deployment_id=deployment_id,
                    file_path=str(absolute_path),
                    file_type="image",
                    file_format=absolute_path.suffix.lstrip(".").lower() if absolute_path.exists() else "jpg",
                    size_bytes=absolute_path.stat().st_size if absolute_path.exists() else None,
                    timestamp=datetime.fromtimestamp(absolute_path.stat().st_mtime) if absolute_path.exists() else datetime.utcnow(),
                    width_px=img.get("width"),
                    height_px=img.get("height"),
                )
                db.add(file_record)
                db.flush()  # Get file_record.id

            # Create Detection records
            for det in img.get("detections", []):
                total_detections += 1

                # Map category
                category_num = det["category"]
                category_map = {"1": "animal", "2": "person", "3": "vehicle"}
                category = category_map.get(category_num, "animal")

                # Count by category
                if category == "animal":
                    animal_count += 1
                elif category == "person":
                    person_count += 1
                elif category == "vehicle":
                    vehicle_count += 1

                # Get bbox
                bbox = det["bbox"]  # [x, y, width, height]

                # Get classifications (if present)
                species = None
                species_confidence = None
                classification_all_probs = None

                if "classifications" in det and det["classifications"]:
                    # Get top classification
                    top_class_id, top_conf = det["classifications"][0]

                    # Get classification_categories mapping
                    class_names = results.get("classification_categories", {})
                    species = class_names.get(str(top_class_id))
                    species_confidence = float(top_conf)

                    # Build all_probabilities dict
                    classification_all_probs = {}
                    for class_id, conf in det["classifications"]:
                        class_name = class_names.get(str(class_id))
                        if class_name:
                            classification_all_probs[class_name] = float(conf)

                    if species:
                        classified_count += 1

                # Create Detection record
                detection_id = det.get("detection_id", str(uuid.uuid4()))

                # Extract frame_number if present (for video detections)
                frame_number = det.get("frame_number")

                detection_data = DetectionCreate(
                    file_id=file_record.id,
                    job_id=job_id,
                    category=category,
                    confidence=float(det["conf"]),
                    bbox_x=float(bbox[0]),
                    bbox_y=float(bbox[1]),
                    bbox_width=float(bbox[2]),
                    bbox_height=float(bbox[3]),
                    frame_number=frame_number,
                )

                detection_record = detection_crud.create_detection(db, detection_data)

                # Update detection with classification data if present
                if species:
                    detection_record.species = species
                    detection_record.species_confidence = species_confidence
                    detection_record.classification_all_probs = classification_all_probs

        # Commit all records
        db.commit()

        logger.info(
            f"Database load complete: {total_detections} detections, "
            f"{classified_count} classified"
        )

        return PipelineResult(
            total_files=len(results.get("images", [])),
            total_detections=total_detections,
            animal_detections=animal_count,
            person_detections=person_count,
            vehicle_detections=vehicle_count,
            classified_detections=classified_count,
        )


# Import BoundingBox at the end to avoid circular import
from app.ml.inference.base import BoundingBox


def load_json_to_database(
    json_path: Path,
    deployment_id: str,
    deployment_folder: Path,
    job_id: str,
    db: Session,
) -> PipelineResult:
    """
    Load JSON file (merged video+image results) to database.

    Standalone function for use by detection_worker.py when processing
    video+image deployments separately. Handles frame_number field for
    video detections.

    Args:
        json_path: Path to JSON file (merged addaxai-run.json)
        deployment_id: Deployment ID
        deployment_folder: Deployment folder (base for relative paths)
        job_id: Job ID
        db: Database session

    Returns:
        PipelineResult with statistics

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        RuntimeError: If database load fails
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        # Load JSON
        with open(json_path) as f:
            results = json.load(f)

        logger.info(f"Loading {len(results.get('images', []))} images/videos to database")

        # Track statistics
        total_detections = 0
        animal_count = 0
        person_count = 0
        vehicle_count = 0
        classified_count = 0

        # Group detections by file
        file_detections = defaultdict(list)

        for img in results.get("images", []):
            relative_file = img["file"]
            absolute_path = (deployment_folder / relative_file).resolve()

            # Determine file type (video or image)
            file_format = absolute_path.suffix.lstrip(".").lower() if absolute_path.exists() else ""
            video_extensions = {"mp4", "avi", "mov", "mkv", "m4v", "wmv", "flv"}
            file_type = "video" if file_format in video_extensions else "image"

            # Get or create File record
            # First check by file_id if provided in JSON
            file_id = img.get("file_id")
            file_record = None

            if file_id:
                file_record = db.query(File).filter(File.id == file_id).first()

            # If not found by file_id, check by file_path AND deployment_id
            # This ensures each project gets its own file records even for the same physical file
            if not file_record:
                file_record = (
                    db.query(File)
                    .filter(File.file_path == str(absolute_path))
                    .filter(File.deployment_id == deployment_id)
                    .first()
                )

            # Create new file record if still not found
            if not file_record:
                if not file_id:
                    file_id = str(uuid.uuid4())

                file_record = File(
                    id=file_id,
                    deployment_id=deployment_id,
                    file_path=str(absolute_path),
                    file_type=file_type,
                    file_format=file_format,
                    size_bytes=absolute_path.stat().st_size if absolute_path.exists() else None,
                    timestamp=datetime.fromtimestamp(absolute_path.stat().st_mtime) if absolute_path.exists() else datetime.utcnow(),
                    width_px=img.get("width"),
                    height_px=img.get("height"),
                )
                db.add(file_record)
                db.flush()  # Get file_record.id

            # Create Detection records
            for det in img.get("detections", []):
                total_detections += 1

                # Map category
                category_num = det["category"]
                category_map = {"1": "animal", "2": "person", "3": "vehicle"}
                category = category_map.get(category_num, "animal")

                # Count by category
                if category == "animal":
                    animal_count += 1
                elif category == "person":
                    person_count += 1
                elif category == "vehicle":
                    vehicle_count += 1

                # Get bbox
                bbox = det["bbox"]  # [x, y, width, height]

                # Get classifications (if present)
                species = None
                species_confidence = None
                classification_all_probs = None

                if "classifications" in det and det["classifications"]:
                    # Get top classification
                    top_class_id, top_conf = det["classifications"][0]

                    # Get classification_categories mapping
                    class_names = results.get("classification_categories", {})
                    species = class_names.get(str(top_class_id))
                    species_confidence = float(top_conf)

                    # Build all_probabilities dict
                    classification_all_probs = {}
                    for class_id, conf in det["classifications"]:
                        class_name = class_names.get(str(class_id))
                        if class_name:
                            classification_all_probs[class_name] = float(conf)

                    if species:
                        classified_count += 1

                # Create Detection record
                detection_id = det.get("detection_id", str(uuid.uuid4()))

                # Extract frame_number if present (for video detections)
                frame_number = det.get("frame_number")

                detection_data = DetectionCreate(
                    file_id=file_record.id,
                    job_id=job_id,
                    category=category,
                    confidence=float(det["conf"]),
                    bbox_x=float(bbox[0]),
                    bbox_y=float(bbox[1]),
                    bbox_width=float(bbox[2]),
                    bbox_height=float(bbox[3]),
                    frame_number=frame_number,
                )

                detection_record = detection_crud.create_detection(db, detection_data)

                # Update detection with classification data if present
                if species:
                    detection_record.species = species
                    detection_record.species_confidence = species_confidence
                    detection_record.classification_all_probs = classification_all_probs

        # Commit all records
        db.commit()

        logger.info(
            f"Database load complete: {total_detections} detections, "
            f"{classified_count} classified"
        )

        return PipelineResult(
            total_files=len(results.get("images", [])),
            total_detections=total_detections,
            animal_detections=animal_count,
            person_detections=person_count,
            vehicle_detections=vehicle_count,
            classified_detections=classified_count,
        )

    except Exception as e:
        logger.error(f"Failed to load JSON to database: {e}", exc_info=True)
        raise RuntimeError(f"Database load failed: {e}") from e
