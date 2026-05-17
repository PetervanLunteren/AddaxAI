"""Privacy-safe copies: blur person and vehicle bounding boxes.

Researchers regularly want to share camera trap output while
protecting bystanders' privacy. This module produces a copy of each
image (or each video's best frame) with every person and vehicle
detection blurred out, so the deliverable can be shared, posted, or
published without exposing identifiable people or licence plates.

Detection routing:

- Categories blurred: `person` and `vehicle`.
- Threshold + verified override: same rule as everywhere else.
  A verified person detection below threshold is still blurred,
  because the human reviewer has explicitly confirmed there is a
  person there.
- Files with zero person / vehicle detections are skipped, not
  copied. Producing identical copies would double disk usage for
  blank or animal-only folders without adding value.

Blur algorithm: per-bbox Gaussian blur with a radius scaled to the
shorter side of the image. 4% of min(w, h) gives a heavy blur on
both small and large images without going pixelly. The blurred
region is pasted back onto the original; everything outside the
bboxes stays sharp.

Copy semantics only. Source files are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageFilter
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.logging_config import get_logger
from app.models import Deployment, Detection, File, Project

from ._exif_writer import ExifBatch, build_tag_set
from ._label_filter import file_is_dropped_by_filter

logger = get_logger(__name__)


# Detection.category values we treat as identifying. Animals are
# always left sharp.
_BLUR_CATEGORIES = ("person", "vehicle")

# Blur radius as a fraction of the image's shorter side. Same value
# the legacy AddaxAI used; testers comparing the old and new outputs
# get a visually identical blur strength.
_BLUR_FRACTION = 0.04
# Floor for very small images so the blur never disappears entirely.
_MIN_BLUR_RADIUS_PX = 8


@dataclass
class BlurPeopleResult:
    """Summary of a blur-people run.

    ``skipped_excluded`` counts animal files whose every passing
    species label was in the user's exclusion set. The user told us
    they want those species absent from all outputs, so we skip
    the blurred copy too.
    """

    written_count: int = 0
    blurred_box_count: int = 0
    skipped_no_target: int = 0
    skipped_missing_source: int = 0
    skipped_excluded: int = 0
    renamed_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "written_count": self.written_count,
            "blurred_box_count": self.blurred_box_count,
            "skipped_no_target": self.skipped_no_target,
            "skipped_missing_source": self.skipped_missing_source,
            "skipped_excluded": self.skipped_excluded,
            "renamed_count": self.renamed_count,
            "errors": list(self.errors),
        }


def _detections_to_blur(
    db: Session,
    file: File,
    threshold: float,
) -> list[Detection]:
    """Return the threshold-aware person / vehicle detections on a file."""
    stmt = (
        select(Detection)
        .where(Detection.file_id == file.id)
        .where(Detection.category.in_(_BLUR_CATEGORIES))
        .where(
            or_(
                Detection.confidence >= threshold,
                Detection.verified == True,  # noqa: E712
            )
        )
        .where(Detection.bbox_x.is_not(None))
    )
    if file.file_type == "video":
        if file.best_frame_number is None:
            return []
        stmt = stmt.where(Detection.frame_number == file.best_frame_number)
    return list(db.execute(stmt).scalars().all())


def _source_for(file: File) -> Path | None:
    """Same source-resolution rule as visualised_images."""
    if file.file_type == "image":
        return Path(file.file_path)
    if file.file_type == "video":
        if not file.best_frame_path:
            return None
        return Path(file.best_frame_path)
    return None


def _destination_name(file: File) -> str:
    """Image keeps its name; video uses stem.jpg (best frame)."""
    source = Path(file.file_path)
    if file.file_type == "video":
        return source.stem + ".jpg"
    return source.name


def _unique_destination(target_dir: Path, source_name: str) -> tuple[Path, bool]:
    """Same collision-suffix logic as the other postprocess modules."""
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix
    candidate = target_dir / source_name
    if not candidate.exists():
        return candidate, False
    counter = 2
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate, True
        counter += 1


def _blur_radius(image: Image.Image) -> int:
    short_side = min(image.size)
    return max(_MIN_BLUR_RADIUS_PX, int(short_side * _BLUR_FRACTION))


def _blur_region(
    image: Image.Image, detection: Detection, radius: int
) -> None:
    """Blur a single bbox in place on the image."""
    if (
        detection.bbox_x is None
        or detection.bbox_y is None
        or detection.bbox_width is None
        or detection.bbox_height is None
    ):
        return

    w, h = image.size
    x0 = max(0, int(detection.bbox_x * w))
    y0 = max(0, int(detection.bbox_y * h))
    x1 = min(w, int((detection.bbox_x + detection.bbox_width) * w))
    y1 = min(h, int((detection.bbox_y + detection.bbox_height) * h))
    if x1 <= x0 or y1 <= y0:
        return

    region = image.crop((x0, y0, x1, y1))
    blurred = region.filter(ImageFilter.GaussianBlur(radius=radius))
    image.paste(blurred, (x0, y0))


def blur_people(
    db: Session,
    project_id: str,
    target_dir: Path,
    *,
    excluded_label_ids: frozenset[str] | None = None,
) -> BlurPeopleResult:
    """Write a privacy-safe copy of every file with a person / vehicle hit.

    Animal files whose every passing label is in
    ``excluded_label_ids`` are skipped — the user wants those species
    absent from all outputs, and the blurred copy is an output.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    target_dir.mkdir(parents=True, exist_ok=True)
    threshold = project.detection_threshold

    files = db.execute(
        select(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    ).scalars().all()

    result = BlurPeopleResult()

    with ExifBatch() as exif_batch:
        for file in files:
            if file_is_dropped_by_filter(
                db, file, threshold, excluded_label_ids
            ):
                result.skipped_excluded += 1
                continue

            targets = _detections_to_blur(db, file, threshold)
            if not targets:
                # Nothing to blur on this file. Skip rather than emit
                # an identical-looking copy.
                result.skipped_no_target += 1
                continue

            source = _source_for(file)
            if source is None or not source.exists():
                result.skipped_missing_source += 1
                continue

            try:
                image = Image.open(source).convert("RGB")
            except OSError as e:
                result.errors.append(f"Could not open {source}: {e}")
                logger.exception(f"blur_people: open failed for {source}")
                continue

            radius = _blur_radius(image)
            for det in targets:
                _blur_region(image, det, radius)
                result.blurred_box_count += 1

            destination, renamed = _unique_destination(
                target_dir, _destination_name(file)
            )
            try:
                image.save(destination)
            except OSError as e:
                result.errors.append(f"Could not save {destination}: {e}")
                logger.exception(
                    f"blur_people: save failed for {destination}"
                )
                continue

            result.written_count += 1
            if renamed:
                result.renamed_count += 1

            # Silent EXIF write on the blurred copy.
            tag_set = build_tag_set(
                db,
                file,
                project,
                APP_VERSION,
                excluded_label_ids=excluded_label_ids,
            )
            if tag_set is not None:
                try:
                    exif_batch.write(destination, tag_set)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"blur_people: EXIF write failed for "
                        f"{destination}: {e}"
                    )

    logger.info(
        f"blur_people: project={project_id} "
        f"written={result.written_count} "
        f"boxes={result.blurred_box_count} "
        f"no_target={result.skipped_no_target} "
        f"missing={result.skipped_missing_source} "
        f"excluded={result.skipped_excluded}"
    )
    return result
