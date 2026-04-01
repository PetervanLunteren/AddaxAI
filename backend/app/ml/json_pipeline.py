"""
JSON pipeline: loads detection/classification results from JSON to database.

Following DEVELOPERS.MD principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere
"""

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.crud import detection as detection_crud
from app.api.schemas.detection import DetectionCreate
from app.core.logging_config import get_logger
from app.ml.inference.base import PipelineResult
from app.models import File
from app.utils.media_dates import extract_video_dates
from app.utils.video_utils import _filename_to_frame_number

logger = get_logger(__name__)


def load_json_to_database(
    json_path: Path,
    deployment_id: str,
    deployment_folder: Path,
    job_id: str,
    db: Session,
    excluded_classes: list[str] | None = None,
    artifacts_folder: Path | None = None,
    taxonomy_csv_path: Path | None = None,
    allowed_taxonomy_keys: frozenset[str] | None = None,
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
        excluded_classes: Optional list of label names to exclude from
            classification results. Excluded labels are zeroed out and
            remaining confidences renormalized before writing to DB.
        artifacts_folder: Project-scoped artifacts folder. If provided,
            video_frames are read from artifacts_folder/video_frames/.

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

        # Build separate ID sets for user exclusion vs non-label skip
        from app.ml.label_exclusion import (
            build_non_label_class_ids,
            build_user_excluded_class_ids,
            filter_and_rollup_classifications,
            should_skip_detection,  # noqa: F401 (re-enable with non-label skip)
        )

        class_categories = results.get("classification_categories", {})
        user_excluded_ids = build_user_excluded_class_ids(
            class_categories, excluded_classes
        )
        non_label_ids = build_non_label_class_ids(class_categories)  # noqa: F841

        # Load taxonomy for exclusion rollup
        taxonomy_lookup = None
        class_id_to_name = None
        if taxonomy_csv_path and taxonomy_csv_path.exists() and user_excluded_ids:
            from app.ml.taxonomic_rollup import load_taxonomy_lookup

            taxonomy_lookup = load_taxonomy_lookup(taxonomy_csv_path)
            class_id_to_name = {str(k): v for k, v in class_categories.items()}

        # Track statistics
        total_detections = 0
        animal_count = 0
        person_count = 0
        vehicle_count = 0
        classified_count = 0
        skipped_non_label = 0

        # Track new exclusion rollup entries for taxonomy persistence
        seen_exclusion_entries: set[str] = set()
        exclusion_new_entries: list[dict] = []

        # Pre-extract video dates using exiftool (single process for all videos)
        video_extensions = {"mp4", "avi", "mov", "mkv", "m4v", "wmv", "flv"}
        video_paths: list[Path] = []
        for img in results.get("images", []):
            abs_path = (deployment_folder / img["file"]).resolve()
            fmt = abs_path.suffix.lstrip(".").lower() if abs_path.exists() else ""
            if fmt in video_extensions:
                video_paths.append(abs_path)
        video_dates = extract_video_dates(video_paths) if video_paths else {}

        # Group detections by file
        defaultdict(list)

        # Check for extracted video frames directory
        _af = artifacts_folder or (deployment_folder / ".addaxai")
        video_frames_dir = _af / "video_frames"
        has_extracted_frames = video_frames_dir.exists()

        for img in results.get("images", []):
            relative_file = img["file"]
            absolute_path = (deployment_folder / relative_file).resolve()

            # Determine file type (video or image)
            file_format = absolute_path.suffix.lstrip(".").lower() if absolute_path.exists() else ""
            is_video = file_format in video_extensions
            file_type = "video" if is_video else "image"

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

                # Extract timestamp: exiftool for videos, EXIF for images, mtime fallback
                exif_metadata = img.get("exif_metadata")
                timestamp = None
                if file_type == "video":
                    timestamp = video_dates.get(absolute_path)
                if timestamp is None:
                    if exif_metadata and "DateTimeOriginal" in exif_metadata:
                        try:
                            timestamp = datetime.strptime(
                                exif_metadata["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
                            )
                        except (ValueError, TypeError):
                            pass
                if timestamp is None:
                    timestamp = (
                        datetime.fromtimestamp(absolute_path.stat().st_mtime)
                        if absolute_path.exists()
                        else datetime.utcnow()
                    )

                # Best frame fields (video only)
                best_frame_number = img.get("best_frame_number")
                best_frame_path = None
                if best_frame_number is not None:
                    # MegaDetector's extract_frames preserves relative dir structure
                    relative_video_path = absolute_path.relative_to(deployment_folder)
                    best_frame_path = str(
                        _af
                        / "video_frames"
                        / relative_video_path
                        / f"frame{best_frame_number:06d}.jpg"
                    )

                # Frame rate (video only) - output by MegaDetector's process_video
                frame_rate = img.get("frame_rate")

                file_record = File(
                    id=file_id,
                    deployment_id=deployment_id,
                    file_path=str(absolute_path),
                    file_type=file_type,
                    file_format=file_format,
                    size_bytes=absolute_path.stat().st_size if absolute_path.exists() else None,
                    timestamp=timestamp,
                    width_px=img.get("width"),
                    height_px=img.get("height"),
                    exif_data=exif_metadata,
                    best_frame_number=best_frame_number,
                    best_frame_path=best_frame_path,
                    frame_rate=frame_rate,
                )
                db.add(file_record)
                db.flush()  # Get file_record.id

            # For video files with extracted frames: create frame File rows
            # and build a mapping from frame_number -> frame File record
            frame_file_map: dict[int, File] = {}
            if is_video and has_extracted_frames:
                # MegaDetector's extract_frames_from_video preserves the
                # relative directory structure from the deployment folder
                relative_video_path = absolute_path.relative_to(deployment_folder)
                frames_subdir = video_frames_dir / relative_video_path

                if frames_subdir.exists():
                    video_timestamp = file_record.timestamp
                    native_frame_rate = img.get("frame_rate") or 30.0

                    # Find all extracted frame JPEGs
                    frame_jpgs = sorted(frames_subdir.glob("frame*.jpg"))

                    # Read dimensions from the first frame JPEG (all frames
                    # from the same video share the same resolution)
                    frame_width = img.get("width")
                    frame_height = img.get("height")
                    if (not frame_width or not frame_height) and frame_jpgs:
                        from PIL import Image as PILImage

                        with PILImage.open(frame_jpgs[0]) as pil_img:
                            frame_width, frame_height = pil_img.size

                    for frame_jpg in frame_jpgs:
                        try:
                            frame_num = _filename_to_frame_number(frame_jpg.name)
                        except ValueError:
                            continue

                        # Compute timestamp offset from video start
                        frame_offset_seconds = frame_num / native_frame_rate
                        frame_timestamp = video_timestamp + timedelta(seconds=frame_offset_seconds)

                        frame_file = File(
                            id=str(uuid.uuid4()),
                            deployment_id=deployment_id,
                            file_path=str(frame_jpg),
                            file_type="frame",
                            file_format="jpg",
                            size_bytes=frame_jpg.stat().st_size if frame_jpg.exists() else None,
                            timestamp=frame_timestamp,
                            width_px=frame_width,
                            height_px=frame_height,
                            frame_rate=native_frame_rate,
                            source_video_id=file_record.id,
                            source_frame_number=frame_num,
                        )
                        db.add(frame_file)
                        frame_file_map[frame_num] = frame_file

                    db.flush()
                    frame_count = len(frame_file_map)
                    logger.debug(
                        f"Created {frame_count} frame records "
                        f"for video {relative_video_path}"
                    )

            # Track categories per-frame (for frame observation_type) and per-video
            frame_categories: dict[int, set[str]] = defaultdict(set)
            video_categories: set[str] = set()

            # Create Detection records
            for det in img.get("detections", []):
                # Map category
                category_num = det["category"]
                category_map = {"1": "animal", "2": "person", "3": "vehicle"}
                category = category_map.get(category_num, "animal")

                # TEMPORARY: disabled non-label skip for SpeciesNet comparison.
                # Re-enable after comparison is complete.
                # if category == "animal" and should_skip_detection(
                #     det, user_excluded_ids, non_label_ids,
                #     class_id_to_name, taxonomy_lookup,
                #     class_categories, allowed_taxonomy_keys,
                # ):
                #     skipped_non_label += 1
                #     continue

                total_detections += 1
                video_categories.add(category)

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
                label = None
                label_confidence = None

                if "classifications" in det and det["classifications"]:
                    classifications = det["classifications"]

                    # Apply user exclusions with rollup (or fallback)
                    if user_excluded_ids:
                        if taxonomy_lookup and class_id_to_name:
                            exclusion_result = (
                                filter_and_rollup_classifications(
                                    classifications,
                                    user_excluded_ids,
                                    class_id_to_name,
                                    taxonomy_lookup,
                                    class_categories,
                                    allowed_taxonomy_keys,
                                    classification_category_descriptions=results.get(
                                        "classification_category_descriptions"
                                    ),
                                )
                            )
                            classifications = exclusion_result.classifications
                            for entry in exclusion_result.new_entries:
                                if entry["name"] not in seen_exclusion_entries:
                                    seen_exclusion_entries.add(entry["name"])
                                    exclusion_new_entries.append(entry)
                        else:
                            from app.ml.label_exclusion import filter_classifications

                            classifications = filter_classifications(
                                classifications, user_excluded_ids
                            )

                    if classifications:
                        # Get top classification
                        top_class_id, top_conf = classifications[0]

                        # Get classification_categories mapping
                        class_names = results.get("classification_categories", {})
                        label = class_names.get(str(top_class_id))
                        label_confidence = float(top_conf)

                    if label:
                        classified_count += 1

                # Create Detection record
                det.get("detection_id", str(uuid.uuid4()))

                # Extract frame_number if present (for video detections)
                frame_number = det.get("frame_number")

                # Map detection to frame File if available, otherwise to video/image File
                detection_file_id = file_record.id
                if frame_number is not None and frame_number in frame_file_map:
                    detection_file_id = frame_file_map[frame_number].id
                    frame_categories[frame_number].add(category)

                detection_data = DetectionCreate(
                    file_id=detection_file_id,
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
                if label:
                    detection_record.label = label
                    detection_record.label_confidence = label_confidence
                    detection_record.classification_method = "machine"
                    # display_name is set by link_detections_to_taxonomy()
                    # which runs after this function in the worker

            # Set observation_type on the video/image File record
            if video_categories:
                if "animal" in video_categories:
                    file_record.observation_type = "animal"
                elif "person" in video_categories:
                    file_record.observation_type = "human"
                elif "vehicle" in video_categories:
                    file_record.observation_type = "vehicle"
            else:
                file_record.observation_type = "blank"

            # Set observation_type on each frame File record
            for frame_num, frame_file in frame_file_map.items():
                cats = frame_categories.get(frame_num, set())
                if "animal" in cats:
                    frame_file.observation_type = "animal"
                elif "person" in cats:
                    frame_file.observation_type = "human"
                elif "vehicle" in cats:
                    frame_file.observation_type = "vehicle"
                else:
                    frame_file.observation_type = "blank"

        # Commit all records
        db.commit()

        logger.info(
            f"Database load complete: {total_detections} detections, "
            f"{classified_count} classified, "
            f"{skipped_non_label} skipped (non-label)"
        )

        return PipelineResult(
            total_files=len(results.get("images", [])),
            total_detections=total_detections,
            animal_detections=animal_count,
            person_detections=person_count,
            vehicle_detections=vehicle_count,
            classified_detections=classified_count,
            exclusion_rollup_entries=exclusion_new_entries,
        )

    except Exception as e:
        logger.error(f"Failed to load JSON to database: {e}", exc_info=True)
        raise RuntimeError(f"Database load failed: {e}") from e
