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

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.crud import detection as detection_crud
from app.api.schemas.detection import DetectionCreate
from app.core.logging_config import get_logger
from app.ml.inference.base import PipelineResult
from app.models import Deployment, File
from app.utils.media_dates import extract_video_dates
from app.utils.video_utils import _filename_to_frame_number

logger = get_logger(__name__)


class MissingTimestampError(RuntimeError):
    """
    Raised only when *every* input file has no extractable capture
    timestamp — i.e. there is nothing to ingest. Partial failures are
    handled by skipping the offending rows and surfacing them through
    `PipelineResult.skipped_missing_timestamp`.

    Observational datetimes are never guessed (no mtime fallback, no
    utcnow substitution). See DEVELOPERS.md "Datetime conventions".
    """

    def __init__(self, missing_paths: list[str]) -> None:
        self.missing_paths = missing_paths
        sample = ", ".join(missing_paths[:5])
        more = f" (+{len(missing_paths) - 5} more)" if len(missing_paths) > 5 else ""
        super().__init__(
            f"No extractable capture timestamp for {len(missing_paths)} file(s): "
            f"{sample}{more}"
        )


def _resolve_capture_timestamp(
    absolute_path: Path,
    *,
    is_video: bool,
    exif_metadata: dict | None,
    video_dates: dict[Path, datetime],
) -> datetime | None:
    """
    Extract the camera's wall-clock capture time for a single file, or
    return None if nothing is available.

    Videos go through exiftool (`video_dates` pre-populated), images
    through MegaDetector's embedded EXIF `DateTimeOriginal`. We never
    substitute a fallback — the caller raises MissingTimestampError for
    any file that returns None here.
    """
    if is_video:
        ts = video_dates.get(absolute_path)
        if ts is not None:
            return ts
    if exif_metadata and "DateTimeOriginal" in exif_metadata:
        try:
            return datetime.strptime(
                exif_metadata["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
            )
        except (ValueError, TypeError):
            return None
    return None


def load_json_to_database(
    json_path: Path,
    deployment_id: str,
    deployment_folder: Path,
    job_id: str,
    db: Session,
    artifacts_folder: Path | None = None,
    taxonomy_name_to_id: (
        dict[str, tuple[str, str | None]] | None
    ) = None,
    builtin_taxonomy_ids: dict[str, str] | None = None,
    datetime_offset_seconds: int = 0,
) -> PipelineResult:
    """
    Load JSON file (merged video+image results) to database.

    Stores raw classifier labels without exclusion or rollup. Phase 7
    (postprocessing) is responsible for applying label exclusion,
    taxonomic rollup, and smoothing as a single unified code path.

    Args:
        json_path: Path to JSON file (merged results.json)
        deployment_id: Deployment ID
        deployment_folder: Deployment folder (base for relative paths)
        job_id: Job ID
        db: Database session
        artifacts_folder: Project-scoped artifacts folder. If provided,
            video_frames are read from artifacts_folder/video_frames/.
        taxonomy_name_to_id: Pre-resolved mapping of
            {lowercase_label: (taxonomy_id, display_name)}.
        builtin_taxonomy_ids: Mapping of builtin category names
            to taxonomy UUIDs, e.g. {"animal": "uuid", ...}.

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

        # Build non-label ID set for skip logic
        from app.ml.label_exclusion import (
            build_non_label_class_ids,
            should_skip_detection,
        )

        class_categories = results.get("classification_categories", {})
        non_label_ids = build_non_label_class_ids(class_categories)

        # Track statistics
        total_detections = 0
        animal_count = 0
        person_count = 0
        vehicle_count = 0
        classified_count = 0
        skipped_non_label = 0

        # Pre-extract video dates using exiftool (single process for all videos)
        video_extensions = {"mp4", "avi", "mov", "mkv", "m4v", "wmv", "flv"}
        video_paths: list[Path] = []
        for img in results.get("images", []):
            abs_path = (deployment_folder / img["file"]).resolve()
            fmt = abs_path.suffix.lstrip(".").lower() if abs_path.exists() else ""
            if fmt in video_extensions:
                video_paths.append(abs_path)
        video_dates = extract_video_dates(video_paths) if video_paths else {}

        # Pre-flight: resolve every image's capture timestamp so the main
        # insert loop can look them up in O(1). Files whose timestamp
        # can't be resolved (corrupted EXIF, missing DateTimeOriginal on
        # an image, exiftool couldn't parse a video header) are recorded
        # in `skipped_missing_timestamp` and skipped during the main
        # loop — same mechanical shape as the existing non-label skip.
        # If *every* input file fails we raise below, because there is
        # nothing left to ingest.
        total_input_images = len(results.get("images", []))
        skipped_missing_timestamp: list[str] = []
        resolved_timestamps: dict[str, datetime] = {}
        for img in results.get("images", []):
            absolute_path = (deployment_folder / img["file"]).resolve()
            fmt = absolute_path.suffix.lstrip(".").lower() if absolute_path.exists() else ""
            ts = _resolve_capture_timestamp(
                absolute_path,
                is_video=fmt in video_extensions,
                exif_metadata=img.get("exif_metadata"),
                video_dates=video_dates,
            )
            if ts is None:
                skipped_missing_timestamp.append(str(absolute_path))
            else:
                resolved_timestamps[str(absolute_path)] = ts
        if (
            total_input_images > 0
            and len(skipped_missing_timestamp) == total_input_images
        ):
            # Nothing left to load — surface this as a hard failure so
            # the worker rolls the placeholder deployment back.
            raise MissingTimestampError(skipped_missing_timestamp)

        # Group detections by file
        defaultdict(list)

        # Check for extracted video frames directory
        _af = artifacts_folder or (deployment_folder / ".addaxai")
        video_frames_dir = _af / "video_frames"
        has_extracted_frames = video_frames_dir.exists()

        for img in results.get("images", []):
            relative_file = img["file"]
            absolute_path = (deployment_folder / relative_file).resolve()

            # Skip files with no extractable capture timestamp. They
            # were recorded in `skipped_missing_timestamp` above and
            # the worker will surface them as a warning on the queue
            # entry. No File row, no detections, no events — same as
            # the existing non-label skip pattern.
            if str(absolute_path) not in resolved_timestamps:
                continue

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

                # Timestamp already resolved in the pre-flight pass.
                captured_at_local = resolved_timestamps[str(absolute_path)]
                exif_metadata = img.get("exif_metadata")

                # Apply user-specified datetime offset (from the "Adjust
                # dates" modal). This corrects camera firmware clock errors
                # like factory resets to 1970 or AM/PM mistakes.
                if datetime_offset_seconds:
                    captured_at_local += timedelta(seconds=datetime_offset_seconds)

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
                    captured_at_local=captured_at_local,
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
                    video_captured_at_local = file_record.captured_at_local
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
                        frame_captured_at_local = (
                            video_captured_at_local + timedelta(seconds=frame_offset_seconds)
                        )

                        frame_file = File(
                            id=str(uuid.uuid4()),
                            deployment_id=deployment_id,
                            file_path=str(frame_jpg),
                            file_type="frame",
                            file_format="jpg",
                            size_bytes=frame_jpg.stat().st_size if frame_jpg.exists() else None,
                            captured_at_local=frame_captured_at_local,
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

                if category == "animal" and should_skip_detection(
                    det, non_label_ids,
                ):
                    skipped_non_label += 1
                    continue

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
                    # Store raw top-1 classification. Exclusion and rollup
                    # are applied in Phase 7 (postprocessing).
                    top_class_id, top_conf = det["classifications"][0]
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
                    # Preserve the raw top-1 classifier output; postprocessing
                    # rollup and user relabels must never touch these columns.
                    detection_record.original_label = label
                    detection_record.original_label_confidence = label_confidence
                    detection_record.classification_method = "machine"
                    # Resolve taxonomy ID and display_name inline
                    if taxonomy_name_to_id:
                        resolved = taxonomy_name_to_id.get(
                            label.lower()
                        )
                        if resolved:
                            detection_record.label_taxonomy_id = (
                                resolved[0]
                            )
                            detection_record.display_name = resolved[1]

                # Set builtin taxonomy ID for unclassified detections
                if not label and builtin_taxonomy_ids:
                    builtin_tid = builtin_taxonomy_ids.get(category)
                    if builtin_tid:
                        detection_record.label_taxonomy_id = builtin_tid
                        detection_record.display_name = (
                            category.capitalize()
                        )

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

        # Derive deployment start/end dates from actual file timestamps
        # so the metadata table shows the real field-deployment window
        # rather than the date the deployment was created.
        min_ts, max_ts = db.execute(
            select(func.min(File.captured_at_local), func.max(File.captured_at_local))
            .where(File.deployment_id == deployment_id)
        ).one()
        if min_ts is not None or max_ts is not None:
            deployment = db.get(Deployment, deployment_id)
            if deployment is not None:
                if min_ts is not None:
                    deployment.start_date_local = min_ts.date()
                deployment.end_date_local = max_ts.date() if max_ts is not None else None
                db.commit()
                logger.info(
                    f"Set deployment {deployment_id} dates from file timestamps: "
                    f"{deployment.start_date_local} to {deployment.end_date_local}"
                )

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
            skipped_missing_timestamp=skipped_missing_timestamp,
        )

    except MissingTimestampError:
        # Propagate as-is so the worker can surface the file list
        # directly; don't bury it inside a generic RuntimeError.
        logger.error("Phase 6 aborted: files missing capture timestamps")
        raise
    except Exception as e:
        logger.error(f"Failed to load JSON to database: {e}", exc_info=True)
        raise RuntimeError(f"Database load failed: {e}") from e
