"""
JSON pipeline: loads detection/classification results from JSON to database,
and shared JSON-level helpers (classification-on-JSON, merge) used by the
deployment worker.

Following DEVELOPERS.MD principles:
- Crash early if setup fails
- Explicit error handling
- Type hints everywhere
"""

import asyncio
import json
import time
import uuid
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
    extract_animal_detections,
)
from app.ml.results_json import iter_images, read_top_level_object
from app.models import Deployment, File
from app.utils.media_dates import (
    extract_video_dates,
    parse_addaxai_filename_datetime,
)

logger = get_logger(__name__)


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
    through MegaDetector's embedded EXIF `DateTimeOriginal`. As a last
    resort, an opt-in `…addaxai-YYYYMMDD-HHMMSS.<ext>` filename is parsed,
    for files whose metadata carries no readable date. A file that still
    returns None is ingested with captured_at_local=NULL and surfaced via
    `PipelineResult.skipped_missing_timestamp`.
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
            pass
    ts = parse_addaxai_filename_datetime(absolute_path.name)
    if ts is not None:
        logger.debug(
            "Capture time from addaxai filename: %s -> %s", absolute_path.name, ts
        )
    return ts


def _safe_file_size(path: Path) -> int | None:
    """File size via a single stat(), or None if unreadable.

    The DB load runs this once per file. Using one stat() instead of
    exists()+stat() halves the per-file filesystem syscalls, which is a major
    share of load time on large deployments stored on slow external / network
    drives (see SIMON_FEEDBACK B2).
    """
    try:
        return path.stat().st_size
    except OSError:
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
            {lowercase_label: (taxonomy_id, scientific_name)}.
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
        # Stream the results JSON instead of json.load: a large deployment's
        # file would otherwise peak at ~6.4x its size in RAM (measured) and
        # OOM the backend. iter_images walks it at flat memory. We make two
        # passes over images (collect video paths, then insert) plus a cheap
        # metadata read; capture timestamps are resolved inline in the second
        # pass rather than in a held-in-memory dict.
        logger.info("Streaming images/videos to database")

        # Build non-label ID set for skip logic
        from app.ml.label_exclusion import (
            build_non_label_class_ids,
            should_skip_detection,
        )

        class_categories = read_top_level_object(
            json_path, "classification_categories"
        )
        non_label_ids = build_non_label_class_ids(class_categories)

        # Track statistics
        total_files = 0
        total_detections = 0
        animal_count = 0
        person_count = 0
        vehicle_count = 0
        classified_count = 0
        skipped_non_label = 0
        skipped_missing_timestamp: list[str] = []
        # MegaDetector failure entries (undecodable video etc.), collected
        # during the insert pass instead of a separate collect_md_failures
        # pass over the whole dict.
        skipped_video_failures: list[dict] = []

        # Pass 1: pre-extract video dates using exiftool (single process for
        # all videos). Skip MegaDetector-failure entries (video could not be
        # decoded; `detections: null`). The worker surfaces those separately
        # as queue warnings; there is no usable file row to create here.
        video_extensions = {"mp4", "avi", "mov", "mkv", "m4v", "wmv", "flv"}
        video_paths: list[Path] = []
        for img in iter_images(json_path):
            if img.get("failure"):
                continue
            abs_path = (deployment_folder / img["file"]).resolve()
            fmt = abs_path.suffix.lstrip(".").lower() if abs_path.exists() else ""
            if fmt in video_extensions:
                video_paths.append(abs_path)
        video_dates = extract_video_dates(video_paths) if video_paths else {}

        # Best-frame JPEGs (one per video) land under this directory.
        # `best_frame_path` on each video File row points into the same
        # tree; legacy data uses the same layout, so the path math works
        # for new and old runs alike.
        _af = artifacts_folder or (deployment_folder / ".addaxai")

        # Pass 2: stream images again and insert. Counting here (before the
        # failure skip) matches the old total_files = len(images).
        for img in iter_images(json_path):
            total_files += 1
            if img.get("failure"):
                skipped_video_failures.append(
                    {"file": img.get("file"), "reason": img.get("failure")}
                )
                continue
            relative_file = img["file"]
            absolute_path = (deployment_folder / relative_file).resolve()

            # Files with no extractable capture timestamp are still
            # ingested (data-agnostic): they get a File row with
            # captured_at_local=NULL, still recorded in
            # `skipped_missing_timestamp` so the UI can surface the count.
            # Time-based stats exclude them; each becomes its own
            # single-file event downstream.

            # Determine file type (video or image)
            file_format = absolute_path.suffix.lstrip(".").lower() if absolute_path.exists() else ""
            is_video = file_format in video_extensions
            file_type = "video" if is_video else "image"

            # Resolve capture timestamp inline (replaces the old pre-pass).
            # Recorded for every loadable image, including ones whose File
            # row already exists, to match the previous behaviour.
            captured_at_local = _resolve_capture_timestamp(
                absolute_path,
                is_video=is_video,
                exif_metadata=img.get("exif_metadata"),
                video_dates=video_dates,
            )
            if captured_at_local is None:
                skipped_missing_timestamp.append(str(absolute_path))

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

                # captured_at_local already resolved inline above.
                exif_metadata = img.get("exif_metadata")

                # Apply user-specified datetime offset (from the "Adjust
                # dates" modal). This corrects camera firmware clock errors
                # like factory resets to 1970 or AM/PM mistakes.
                if datetime_offset_seconds and captured_at_local is not None:
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

                # Image dimensions. MD writes `width`/`height` for images
                # but `process_video` does not, so video entries arrive
                # without them. Backfill from the best-frame JPEG (which
                # the classifier worker or the no-classifier streaming
                # pass has already written by the time we reach DB load).
                # Without this `_compute_crop_bbox` returns None and the
                # observations grid renders crops with no bbox overlay.
                width_px = img.get("width")
                height_px = img.get("height")
                if is_video and (not width_px or not height_px) and best_frame_path:
                    bf = Path(best_frame_path)
                    if bf.is_file():
                        try:
                            from PIL import Image as PILImage

                            with PILImage.open(bf) as bf_img:
                                width_px, height_px = bf_img.size
                        except Exception as e:
                            logger.warning(
                                f"Could not read dims from {bf}: {e}"
                            )

                file_record = File(
                    id=file_id,
                    deployment_id=deployment_id,
                    file_path=str(absolute_path),
                    file_type=file_type,
                    file_format=file_format,
                    size_bytes=_safe_file_size(absolute_path),
                    captured_at_local=captured_at_local,
                    width_px=width_px,
                    height_px=height_px,
                    exif_data=exif_metadata,
                    best_frame_number=best_frame_number,
                    best_frame_path=best_frame_path,
                    frame_rate=frame_rate,
                )
                db.add(file_record)
                db.flush()  # Get file_record.id

            # Video detections live on the parent video File row and
            # keep their `frame_number` column. We no longer create one
            # File row per detection-bearing frame; the disk used to
            # back those rows is gone too (the classifier worker now
            # streams frames straight from the source video).
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
                    label = class_categories.get(str(top_class_id))
                    label_confidence = float(top_conf)

                    if label:
                        classified_count += 1

                # Create Detection record
                det.get("detection_id", str(uuid.uuid4()))

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
                if label:
                    detection_record.label = label
                    detection_record.label_confidence = label_confidence
                    # Preserve the raw top-1 classifier output; postprocessing
                    # rollup and user relabels must never touch these columns.
                    detection_record.original_label = label
                    detection_record.original_label_confidence = label_confidence
                    detection_record.classification_method = "machine"
                    # Resolve taxonomy ID + both names inline
                    if taxonomy_name_to_id:
                        resolved = taxonomy_name_to_id.get(
                            label.lower()
                        )
                        if resolved:
                            detection_record.label_taxonomy_id = (
                                resolved[0]
                            )
                            detection_record.scientific_name = resolved[1]
                            detection_record.common_name = resolved[2]

                # Set builtin taxonomy ID for unclassified detections
                if not label and builtin_taxonomy_ids:
                    builtin_tid = builtin_taxonomy_ids.get(category)
                    if builtin_tid:
                        detection_record.label_taxonomy_id = builtin_tid
                        detection_record.scientific_name = (
                            category.capitalize()
                        )
                        detection_record.common_name = (
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
            total_files=total_files,
            total_detections=total_detections,
            animal_detections=animal_count,
            person_detections=person_count,
            vehicle_detections=vehicle_count,
            classified_detections=classified_count,
            skipped_missing_timestamp=skipped_missing_timestamp,
            skipped_video_failures=skipped_video_failures,
        )

    except Exception as e:
        logger.error(f"Failed to load JSON to database: {e}", exc_info=True)
        raise RuntimeError(f"Database load failed: {e}") from e


def load_json_to_database_owned_session(
    json_path: Path,
    deployment_id: str,
    deployment_folder: Path,
    job_id: str,
    artifacts_folder: Path | None = None,
    taxonomy_name_to_id: (
        dict[str, tuple[str, str | None]] | None
    ) = None,
    builtin_taxonomy_ids: dict[str, str] | None = None,
    datetime_offset_seconds: int = 0,
) -> PipelineResult:
    """Run load_json_to_database with a session created in THIS thread.

    The per-file insert loop is the heavy, event-loop-blocking part of the save
    phase. The async detection worker runs this via ``asyncio.to_thread`` so the
    backend stays responsive (delete deployment, add to queue) during big runs
    instead of hanging (see SIMON_FEEDBACK B2/B3/B4). SQLite's default
    ``check_same_thread`` requires the session be created in the worker thread,
    so this owns a fresh session rather than taking the caller's. Taxonomy rows
    are already committed by the caller and passed in as plain dicts, so this
    session only inserts files/detections and commits them.
    """
    from app.db.base import get_session_factory

    db = get_session_factory()()
    try:
        return load_json_to_database(
            json_path=json_path,
            deployment_id=deployment_id,
            deployment_folder=deployment_folder,
            job_id=job_id,
            db=db,
            artifacts_folder=artifacts_folder,
            taxonomy_name_to_id=taxonomy_name_to_id,
            builtin_taxonomy_ids=builtin_taxonomy_ids,
            datetime_offset_seconds=datetime_offset_seconds,
        )
    finally:
        db.close()


async def run_classification_on_json(
    json_path: Path,
    classification_model,
    deployment_folder: Path,
    batch_size: int,
    progress_callback: Callable[[str, float, dict | None], None] | None = None,
    classification_model_dir: Path | None = None,
    best_frame_output_base: Path | None = None,
    job_id: str | None = None,
) -> None:
    """
    Run classification on a detection JSON file.

    Updates the JSON file in-place with classification results. Also picks
    a `best_frame_number` for every video in the JSON during the same
    streaming pass (the classifier worker scores sharpness while it has
    the frames in memory) and stamps it onto the per-image entry. Pure
    JSON operation, no database access. Used by the deployment worker.

    Args:
        json_path: Path to detection JSON file
        classification_model: Classification model instance
        deployment_folder: Folder used to resolve relative file paths
            in the JSON
        batch_size: Number of crops per classification batch
        progress_callback: Optional progress callback
        classification_model_dir: Path to classification model directory
            (for taxonomy.csv)
        best_frame_output_base: Directory under which the worker drops
            one best-frame JPEG per video, mirroring the relative video
            path. If None, falls back to
            `deployment_folder/.addaxai/video_frames` (the legacy layout
            so `best_frame_path` math in `_load_to_database` keeps
            working unchanged).

    Raises:
        RuntimeError: If classification fails
    """
    logger.info("Running per-detection classification")

    with open(json_path) as f:
        md_results = json.load(f)

    animal_detections = extract_animal_detections(md_results)

    # Build the best-frame output map up front: every non-failed video
    # in the JSON gets a destination directory, including blank videos
    # so we still produce a thumbnail for them.
    _bf_base = best_frame_output_base or (
        deployment_folder / ".addaxai" / "video_frames"
    )
    best_frame_outputs: dict[str, str] = {}
    video_path_by_abs: dict[str, Path] = {}
    for img_info in md_results.get("images", []) or []:
        if img_info.get("failure"):
            continue
        file_path = (deployment_folder / img_info["file"]).resolve()
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative_video_path = file_path.relative_to(deployment_folder)
        dest_dir = _bf_base / relative_video_path
        best_frame_outputs[str(file_path)] = str(dest_dir)
        video_path_by_abs[str(file_path)] = file_path

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
            items.append({
                "source": "video",
                "video_path": str(file_path),
                "frame_number": int(frame_number),
                "bbox": detection["bbox"],
                "detection_conf": float(detection.get("conf", 0.0)),
            })
        else:
            if not file_path.exists():
                logger.warning(f"Image not found: {file_path}, skipping")
                continue
            items.append({
                "source": "image",
                "image_path": str(file_path),
                "bbox": detection["bbox"],
            })
        indices.append((img_idx, det_idx))

    video_items = sum(1 for it in items if it["source"] == "video")
    image_items = len(items) - video_items
    total_animals = len(animal_detections)
    logger.info(
        f"[DEBUG] Built {len(items)} items for batch classification "
        f"({image_items} image crops, {video_items} video-frame crops) "
        f"plus {len(best_frame_outputs)} best-frame targets"
    )

    if not items and not best_frame_outputs:
        logger.info("No valid items to classify and no best frames to score")
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
        # Surface a clean status caption while the classification
        # subprocess loads its model. Without this, the frontend sits
        # on a generic "Starting up..." for the duration of the model
        # load (5-15s for SpeciesNet) until the first tqdm tick arrives.
        if progress_callback:
            sync_cls_progress("Loading classification model...", 0.0, None)
        results, class_names, compute_device, best_frames = (
            classification_model.classify_detections(
                items,
                best_frame_outputs=best_frame_outputs,
                batch_size=batch_size,
                progress_callback=on_progress,
                job_id=job_id,
            )
        )
        logger.info(
            f"[DEBUG] classify_detections() returned: "
            f"{len(results)} results, {len(class_names)} classes, "
            f"{len(best_frames)} best frames, device={compute_device}"
        )

        if progress_callback and compute_device:
            sync_cls_progress("Classifying...", 1.0, {"compute_device": compute_device})

        # Stamp best_frame_number onto each video's image entry. Map by
        # absolute file path because the worker keyed its output that way.
        if best_frames:
            abs_to_img_idx: dict[str, int] = {}
            for img_idx, img_info in enumerate(md_results.get("images", []) or []):
                if img_info.get("failure"):
                    continue
                abs_path = str((deployment_folder / img_info["file"]).resolve())
                abs_to_img_idx[abs_path] = img_idx
            for video_path, best_fn in best_frames.items():
                idx = abs_to_img_idx.get(video_path)
                if idx is None:
                    continue
                md_results["images"][idx]["best_frame_number"] = int(best_fn)

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

        # Write metadata (categories, info) before the big images array so a
        # streaming reader can grab classification_categories without scanning
        # the whole file. Moving images to the end of the insertion order does
        # this; key order is not semantically meaningful to any consumer.
        merged_data["images"] = merged_data.pop("images")

        with open(output_file, "w") as f:
            json.dump(merged_data, f, indent=2)

        logger.info(f"Merged {len(json_files)} JSON files to {output_file}")

    except Exception as e:
        logger.error(f"JSON merge failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to merge JSON files: {e}") from e
