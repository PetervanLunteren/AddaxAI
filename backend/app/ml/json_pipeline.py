"""
JSON pipeline: loads detection/classification results from JSON to database,
and shared JSON-level helpers (classification-on-JSON, merge) used by both
the deployment worker and the Timelapse runner.

Following DEVELOPERS.MD principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.crud import detection as detection_crud
from app.api.schemas.detection import DetectionCreate
from app.core.logging_config import get_logger
from app.core.media_types import VIDEO_EXTENSIONS
from app.ml.inference.base import PipelineResult
from app.ml.json_utils import (
    build_classification_category_descriptions,
    collect_md_failures,
    extract_animal_detections,
)
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
        # Skip MegaDetector-failure entries (video could not be decoded;
        # `detections: null`). The worker surfaces those separately as
        # queue warnings; there is no usable file row to create here.
        video_extensions = {"mp4", "avi", "mov", "mkv", "m4v", "wmv", "flv"}
        video_paths: list[Path] = []
        for img in results.get("images") or []:
            if img.get("failure"):
                continue
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
        loadable_images = [
            img for img in (results.get("images") or []) if not img.get("failure")
        ]
        total_input_images = len(loadable_images)
        skipped_missing_timestamp: list[str] = []
        resolved_timestamps: dict[str, datetime] = {}
        for img in loadable_images:
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

        for img in loadable_images:
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

            # Create Detection records. `detections or []` keeps the loop
            # safe even if `loadable_images` filtering above is bypassed
            # by future callers passing in raw process_video output.
            for det in img.get("detections") or []:
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
            total_files=len(results.get("images") or []),
            total_detections=total_detections,
            animal_detections=animal_count,
            person_detections=person_count,
            vehicle_detections=vehicle_count,
            classified_detections=classified_count,
            skipped_missing_timestamp=skipped_missing_timestamp,
            skipped_video_failures=collect_md_failures(results),
        )

    except MissingTimestampError:
        # Propagate as-is so the worker can surface the file list
        # directly; don't bury it inside a generic RuntimeError.
        logger.error("Phase 6 aborted: files missing capture timestamps")
        raise
    except Exception as e:
        logger.error(f"Failed to load JSON to database: {e}", exc_info=True)
        raise RuntimeError(f"Database load failed: {e}") from e


async def run_classification_on_json(
    json_path: Path,
    classification_model,
    deployment_folder: Path,
    batch_size: int,
    progress_callback: Callable[[str, float, dict | None], None] | None = None,
    classification_model_dir: Path | None = None,
    video_frames_base_dir: Path | None = None,
    job_id: str | None = None,
) -> None:
    """
    Run classification on a detection JSON file.

    Updates the JSON file in-place with classification results. Pure JSON
    operation, no database access. Reused by both the deployment worker
    and the Timelapse runner.

    Args:
        json_path: Path to detection JSON file
        classification_model: Classification model instance
        deployment_folder: Folder used to resolve relative file paths
            in the JSON
        batch_size: Number of crops per classification batch
        progress_callback: Optional progress callback
        classification_model_dir: Path to classification model directory
            (for taxonomy.csv)
        video_frames_base_dir: Path to video_frames directory. If None,
            falls back to deployment_folder / ".addaxai" / "video_frames"

    Raises:
        RuntimeError: If classification fails
    """
    logger.info("Running per-detection classification")

    with open(json_path) as f:
        md_results = json.load(f)

    animal_detections = extract_animal_detections(md_results)
    total_animals = len(animal_detections)

    if total_animals == 0:
        logger.info("No animals to classify")
        return

    items: list[dict] = []
    indices: list[tuple[int, int]] = []

    for img_idx, det_idx, detection in animal_detections:
        img_info = md_results["images"][img_idx]
        relative_file = img_info["file"]
        file_path = (deployment_folder / relative_file).resolve()

        is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS

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
            job_id=job_id,
        )
        logger.info(
            f"[DEBUG] classify_detections() returned: "
            f"{len(results)} results, {len(class_names)} classes, device={compute_device}"
        )

        if progress_callback and compute_device:
            sync_cls_progress("Classifying...", 1.0, {"compute_device": compute_device})

        name_to_id = {name: class_id for class_id, name in class_names.items()}
        classified_count = 0

        for (img_idx, det_idx), result in zip(indices, results, strict=True):
            if result is None:
                continue

            # Store all results (not truncated) so label exclusion can find
            # included labels even if they rank low.
            md_results["images"][img_idx]["detections"][det_idx]["classifications"] = [
                [name_to_id[class_name], prob]
                for class_name, prob in result.all_probabilities.items()
                if class_name in name_to_id
            ]
            classified_count += 1

        if class_names:
            md_results["classification_categories"] = class_names

            if classification_model_dir:
                taxonomy_csv = classification_model_dir / "taxonomy.csv"
                if taxonomy_csv.exists():
                    descriptions = build_classification_category_descriptions(
                        class_names, taxonomy_csv
                    )
                    if descriptions:
                        md_results["classification_category_descriptions"] = descriptions

        with open(json_path, "w") as f:
            json.dump(md_results, f, indent=2)

        logger.info(f"Classified {classified_count}/{total_animals} animals")
        logger.info(
            f"[DEBUG] Wrote updated JSON to {json_path}, "
            f"has classification_categories={bool(md_results.get('classification_categories'))}"
        )

    await loop.run_in_executor(None, _run_batch_classification)


def merge_json_files(
    json_files: list[Path],
    output_file: Path,
    deployment_id: str,
    detection_model_id: str | None = None,
    classification_model_id: str | None = None,
) -> None:
    """
    Merge multiple JSON files (video and image results) into a single file.

    Creates a unified classification_categories mapping and renumbers all
    classification IDs to be consistent across video and image detections.

    This is necessary because video and image JSONs may have different ID
    mappings for the same label. This function unifies the mappings so all
    IDs are consistent.

    Args:
        json_files: List of JSON file paths to merge
        output_file: Output merged JSON file path
        deployment_id: Deployment ID for the info.addaxai metadata block
        detection_model_id: Detection model ID (for info section)
        classification_model_id: Classification model ID (for info section)

    Raises:
        RuntimeError: If merge fails
    """
    try:
        merged_data: dict = {
            "images": [],
            "detection_categories": {},
            "classification_categories": {},
            "classification_category_descriptions": {},
            "info": {},
        }

        # Track unified classification mapping: label_name -> unified_id
        unified_class_mapping: dict[str, str] = {}
        next_class_id = 1

        for json_file in json_files:
            if not json_file.exists():
                logger.warning(f"JSON file not found: {json_file}")
                continue

            with open(json_file) as f:
                data = json.load(f)

            file_class_categories = data.get("classification_categories", {})

            id_remapping: dict[str, str] = {}

            for old_id, label_name in file_class_categories.items():
                if label_name not in unified_class_mapping:
                    unified_class_mapping[label_name] = str(next_class_id)
                    next_class_id += 1

                id_remapping[old_id] = unified_class_mapping[label_name]

            file_descriptions = data.get("classification_category_descriptions", {})
            for old_id, desc_str in file_descriptions.items():
                new_id = id_remapping.get(old_id, old_id)
                merged_data["classification_category_descriptions"][new_id] = desc_str

            for img in data.get("images") or []:
                # Iterate `detections or []` so the merge survives any
                # process_video failure entries that snuck through with
                # `detections: null`.
                for det in img.get("detections") or []:
                    if "classifications" in det and det["classifications"]:
                            renumbered_classifications = []
                            for class_id, confidence in det["classifications"]:
                                old_id_str = str(class_id)
                                if old_id_str in id_remapping:
                                    new_id = id_remapping[old_id_str]
                                    renumbered_classifications.append([new_id, confidence])
                                else:
                                    logger.warning(
                                        f"Unknown classification ID "
                                        f"'{class_id}' in "
                                        f"{json_file.name}, "
                                        f"keeping original"
                                    )
                                    renumbered_classifications.append([class_id, confidence])

                            det["classifications"] = renumbered_classifications

            merged_data["images"].extend(data.get("images") or [])

            if not merged_data["detection_categories"]:
                merged_data["detection_categories"] = data.get("detection_categories", {})
            if not merged_data["info"]:
                merged_data["info"] = data.get("info", {})

        merged_data["classification_categories"] = {
            class_id: label_name for label_name, class_id in unified_class_mapping.items()
        }

        if not merged_data["classification_category_descriptions"]:
            del merged_data["classification_category_descriptions"]

        num_labels = len(merged_data['classification_categories'])
        logger.info(
            f"Unified classification mapping: "
            f"{num_labels} labels "
            f"across {len(json_files)} JSON files"
        )

        addaxai_info: dict = {
            "version": "todo-not-implemented-yet",
            "deployment_id": deployment_id,
            "classification_completion_time": (
                datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            ),
        }
        if detection_model_id:
            addaxai_info["detection_model"] = detection_model_id
        if classification_model_id:
            addaxai_info["classification_model"] = classification_model_id
        merged_data["info"]["addaxai"] = addaxai_info

        with open(output_file, "w") as f:
            json.dump(merged_data, f, indent=2)

        logger.info(f"Merged {len(json_files)} JSON files to {output_file}")

    except Exception as e:
        logger.error(f"JSON merge failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to merge JSON files: {e}") from e
