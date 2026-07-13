"""Combined per-file pass: blur people / vehicles, draw detection boxes.

This module replaces the old ``visualised_images`` + ``blur_people``
pair. Doing both effects in one open / save round-trip means a user
who picks both gets one image per source (legacy AddaxAI behaviour),
not two parallel trees, and writes land directly into the file's
post-separation destination(s) instead of a siloed wrapper folder.

Effect composition (applies in this order):

1. ``anonymise``: blur every person / vehicle bounding box on the
   underlying RGB pixels in place. Privacy-safe: the bystander's face
   or licence plate is gone before the box is drawn over it.
2. ``draw_bboxes``: composite a translucent rounded-box overlay with
   a pill label per detection on top of the (possibly blurred) image.

Destination resolution (via ``OutputContext``):

- Separation ran: write to every path the file was placed at. A
  multi-species file lands in N label folders, so the annotated image
  ends up in N folders too. Videos write the annotated best frame to
  ``<sibling>.jpg`` next to each video destination.
- Separation did not run: write to a fresh destination under
  ``output_root`` (``<original_name>`` for images, ``<stem>.jpg`` for
  videos). Collision-suffixed to avoid clobbering an existing file.

EXIF: every saved destination gets the detection tag set written via
the shared ``ExifBatch``. PIL save strips EXIF from the source, so
this module writes it back on its own outputs whether or not
separation already wrote the same tags on the (now-overwritten)
separated copy.

Source-vs-destination semantics:

- Images: source is ``File.file_path`` (post-separation for ``move``
  mode, original for ``copy`` / no-separation). Effects are computed
  on the source pixels and saved to the destination.
- Videos: source is the per-video best-frame JPEG
  (``File.best_frame_path``). That path is unaffected by separation
  (the best frame lives inside ``.addaxai/projects/...``). The
  annotated JPEG is saved alongside the video destination, NOT
  written into the video container itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.confidence import format_confidence_pct
from app.core.logging_config import get_logger
from app.models import Deployment, Detection, File, Project

from ._exif_writer import ExifBatch, build_tag_set, is_image_path
from ._label_filter import file_is_dropped_by_filter
from ._output_context import OutputContext
from ._visualisation_style import (
    PILL_BG_RGBA,
    STROKE_ALPHA,
    WHITE,
    RenderMetrics,
    detection_color,
    render_metrics,
)

logger = get_logger(__name__)


# Detection.category values treated as identifying for the anonymise
# pass. Animals stay sharp.
_BLUR_CATEGORIES = ("person", "vehicle")

# Blur radius as a fraction of the image's shorter side. Same value
# legacy AddaxAI used so testers comparing outputs see an identical
# blur strength.
_BLUR_FRACTION = 0.04
# Floor for very small images so the blur never disappears entirely.
_MIN_BLUR_RADIUS_PX = 8


@dataclass
class AnnotatedCopiesResult:
    """Summary of an annotated-copies run.

    ``written_count`` counts placements (one per saved destination),
    so a multi-species file under separation contributes once per
    label folder. ``bbox_count`` and ``blurred_box_count`` are
    per-source totals (boxes drawn / blurred); they do not multiply
    with placement count because the same set of detections produces
    every copy.
    """

    written_count: int = 0
    bbox_count: int = 0
    blurred_box_count: int = 0
    skipped_no_change: int = 0
    skipped_missing_source: int = 0
    skipped_excluded: int = 0
    renamed_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "written_count": self.written_count,
            "bbox_count": self.bbox_count,
            "blurred_box_count": self.blurred_box_count,
            "skipped_no_change": self.skipped_no_change,
            "skipped_missing_source": self.skipped_missing_source,
            "skipped_excluded": self.skipped_excluded,
            "renamed_count": self.renamed_count,
            "errors": list(self.errors),
        }


# ─────────────────────────────────────────────────────────────────
# Source / destination resolution
# ─────────────────────────────────────────────────────────────────


def _source_for(file: File) -> Path | None:
    """The on-disk file to read pixels from.

    Images use their own path (which is the post-separation location
    under ``move`` mode, the original under ``copy`` / no separation).
    Videos use the pre-rendered best-frame JPEG, which sits in the
    project's ``.addaxai`` cache and is unaffected by separation.
    """
    if file.file_type == "image":
        return Path(file.file_path)
    if file.file_type == "video":
        if not file.best_frame_path:
            return None
        return Path(file.best_frame_path)
    return None


def _fallback_destination_name(file: File) -> str:
    """Filename for the no-separation case. Image keeps its name;
    video's annotated best frame uses ``<stem>_still.jpg``."""
    if file.file_type == "video":
        from .separate_folders import video_still_name

        return video_still_name(file.file_path)
    return Path(file.file_path).name


def _unique_destination(target_dir: Path, name: str) -> tuple[Path, bool]:
    """Collision-safe destination under ``target_dir``. Appends
    ``_2``, ``_3``, ... until the name is free. Returns the path
    plus a flag indicating whether the rename happened."""
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = target_dir / name
    if not candidate.exists():
        return candidate, False
    counter = 2
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate, True
        counter += 1


def _video_jpeg_sibling(video_path: Path) -> Path:
    """For a separated video at ``<dir>/video.mp4`` we want the
    annotated best frame at ``<dir>/video.jpg``. The video destination
    already has a unique name (separation collision-suffixed it), so
    swapping the suffix produces a unique JPEG sibling too."""
    return video_path.with_suffix(".jpg")


def _resolve_destinations(
    file: File,
    ctx: OutputContext,
    result: AnnotatedCopiesResult,
) -> list[Path]:
    """Where to write annotated copies for one file.

    Returns the post-separation placements (one per label folder),
    or a single freshly allocated path under ``output_root`` when
    separation did not place the file. Videos always end up at a
    ``.jpg`` sibling of the video destination.
    """
    resolved = ctx.resolved_for(file.id)
    if resolved:
        if file.file_type == "video":
            return [_video_jpeg_sibling(p) for p in resolved]
        return resolved

    name = _fallback_destination_name(file)
    dest, renamed = _unique_destination(ctx.output_root, name)
    if renamed:
        result.renamed_count += 1
    return [dest]


# ─────────────────────────────────────────────────────────────────
# Blur pass (anonymise)
# ─────────────────────────────────────────────────────────────────


def _detections_to_blur(
    db: Session, file: File, threshold: float
) -> list[Detection]:
    """Person / vehicle detections to blur on this file.

    Threshold + verified override matches the rule used everywhere
    else: a verified person below threshold is still blurred — the
    human reviewer confirmed it.
    """
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
        stmt = stmt.where(
            Detection.frame_number == file.best_frame_number
        )
    return list(db.execute(stmt).scalars().all())


def _blur_radius(image: Image.Image) -> int:
    short_side = min(image.size)
    return max(_MIN_BLUR_RADIUS_PX, int(short_side * _BLUR_FRACTION))


def _blur_region(
    image: Image.Image, detection: Detection, radius: int
) -> None:
    """Blur a single bbox in place on ``image``."""
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


# ─────────────────────────────────────────────────────────────────
# Bbox + pill drawing pass
# ─────────────────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.ImageFont:
    """Scalable default font at ``size`` px, cached per size.

    Pillow 10+ ``load_default(size=...)`` returns a TrueType-backed
    font that scales cleanly; older Pillow ignores the argument and
    returns the small bitmap font (acceptable fallback)."""
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, AttributeError):  # Pillow < 10
        return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    """Pillow-version-agnostic text width."""
    if hasattr(draw, "textbbox"):
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        return right - left
    return draw.textsize(text, font=font)[0]  # type: ignore[attr-defined]


@dataclass
class _PillLayout:
    category_text: str
    label_text: str
    has_label: bool
    pill_width: int
    pill_height: int
    color: tuple[int, int, int]


def _pill_name(detection: Detection, name_mode: str) -> str:
    """Resolve the species name for the burned-in pill under the active
    display preference (common vs scientific). Mirrors the frontend
    ``resolveSpeciesName`` fallback order so the saved image matches what
    the UI shows."""
    common = detection.common_name or None
    scientific = detection.scientific_name or None
    label = detection.label or None
    ordered = (
        [scientific, common, label]
        if name_mode == "scientific"
        else [common, scientific, label]
    )
    return next((v for v in ordered if v), detection.label or "")


def _compute_pill_layout(
    draw: ImageDraw.ImageDraw,
    detection: Detection,
    m: RenderMetrics,
    font,
    name_mode: str,
) -> _PillLayout:
    """Pill geometry for one detection, sized by ``m``. Two-line when a
    species label is present, single-line otherwise. Both lines share
    one font."""
    color = detection_color(detection.label, detection.category)
    has_label = bool(detection.label)
    category_text = (
        f"{detection.category.capitalize()} "
        f"{format_confidence_pct(detection.confidence)}"
    )
    if has_label:
        species_name = _pill_name(detection, name_mode)
        label_text = (
            f"{species_name[:1].upper()}{species_name[1:]} "
            f"{format_confidence_pct(detection.label_confidence or detection.confidence)}"
        )
        pill_height = (
            m.pad_y + m.font + m.line_gap + m.font + m.pad_y
        )
        w1 = _text_width(draw, category_text, font)
        w2 = _text_width(draw, label_text, font)
        pill_width = m.text_start_x + max(w1, w2) + m.pad_x
    else:
        label_text = ""
        pill_height = m.pad_y + m.font + m.pad_y
        tw = _text_width(draw, category_text, font)
        pill_width = m.text_start_x + tw + m.pad_x
    return _PillLayout(
        category_text=category_text,
        label_text=label_text,
        has_label=has_label,
        pill_width=pill_width,
        pill_height=pill_height,
        color=color,
    )


def _draw_one(
    draw: ImageDraw.ImageDraw,
    detection: Detection,
    image_size: tuple[int, int],
    m: RenderMetrics,
    font,
    name_mode: str,
) -> None:
    """Draw one bbox + pill onto the RGBA overlay, sized by ``m``."""
    if (
        detection.bbox_x is None
        or detection.bbox_y is None
        or detection.bbox_width is None
        or detection.bbox_height is None
    ):
        return
    img_w, img_h = image_size
    x0 = max(0, int(detection.bbox_x * img_w))
    y0 = max(0, int(detection.bbox_y * img_h))
    x1 = min(img_w, int((detection.bbox_x + detection.bbox_width) * img_w))
    y1 = min(img_h, int((detection.bbox_y + detection.bbox_height) * img_h))
    if x1 <= x0 or y1 <= y0:
        return

    color = detection_color(detection.label, detection.category)

    # Single near-solid coloured rounded outline.
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=m.radius,
        outline=(*color, STROKE_ALPHA),
        width=m.stroke,
    )

    pill = _compute_pill_layout(draw, detection, m, font, name_mode)
    pill_x = max(0, min(x0, img_w - pill.pill_width))
    pill_y = (
        y0 - pill.pill_height if y0 - pill.pill_height >= 0 else y0
    )

    draw.rounded_rectangle(
        (pill_x, pill_y, pill_x + pill.pill_width, pill_y + pill.pill_height),
        radius=round(m.radius * 0.9),
        fill=PILL_BG_RGBA,
    )

    # Both lines share one font, full white. Two lines when a species
    # label is present, one (the category) otherwise.
    text_x = pill_x + m.text_start_x
    draw.text(
        (text_x, pill_y + m.pad_y),
        pill.category_text,
        fill=(*WHITE, 255),
        font=font,
    )
    if pill.has_label:
        draw.text(
            (text_x, pill_y + m.pad_y + m.font + m.line_gap),
            pill.label_text,
            fill=(*WHITE, 255),
            font=font,
        )


def _detections_to_draw(
    db: Session,
    file: File,
    threshold: float,
    excluded_label_ids: frozenset[str] | None,
) -> list[Detection]:
    """Detections to draw bboxes for.

    Images: every threshold-or-verified detection with a bbox.
    Videos: only detections anchored to the best frame.
    Excluded species labels are filtered out so visualised copies do
    not show boxes for taxa the user removed.
    """
    stmt = (
        select(Detection)
        .where(Detection.file_id == file.id)
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
        stmt = stmt.where(
            Detection.frame_number == file.best_frame_number
        )
    detections = list(db.execute(stmt).scalars().all())
    if excluded_label_ids:
        detections = [
            d for d in detections
            if not (
                (d.label_taxonomy_id and d.label_taxonomy_id in excluded_label_ids)
                or (d.label and d.label in excluded_label_ids)
            )
        ]
    return detections


# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────


def write_annotated_copies(
    db: Session,
    project_id: str,
    ctx: OutputContext,
    *,
    media_threshold: float,
    draw_bboxes: bool,
    anonymise: bool,
    excluded_label_ids: frozenset[str] | None = None,
    name_mode: str = "common",
) -> AnnotatedCopiesResult:
    """Apply the requested per-file effects and write the result to
    every destination the file lives at under ``ctx``.

    ``media_threshold`` is the Save step's media-output confidence:
    only detections at or above it (or verified) get drawn / blurred,
    matching the separation module's placement rule.

    At least one of ``draw_bboxes`` / ``anonymise`` must be true;
    otherwise the worker should not have invoked the module at all.
    A file with nothing to draw AND nothing to blur (given the
    selected effects and the threshold + exclusion filters) is
    skipped — no point in producing an identical copy.
    """
    if not draw_bboxes and not anonymise:
        raise ValueError(
            "write_annotated_copies needs at least one of "
            "draw_bboxes / anonymise to be true"
        )

    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id!r} not found")

    threshold = media_threshold

    files = db.execute(
        select(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    ).scalars().all()

    result = AnnotatedCopiesResult()

    with ExifBatch() as exif_batch:
        for file in files:
            # Same file-level exclusion the other modules use, so the
            # whole save pipeline agrees on which files are dropped.
            if file_is_dropped_by_filter(
                db, file, threshold, excluded_label_ids
            ):
                result.skipped_excluded += 1
                continue

            blur_targets = (
                _detections_to_blur(db, file, threshold)
                if anonymise
                else []
            )
            bbox_dets = (
                _detections_to_draw(
                    db, file, threshold, excluded_label_ids
                )
                if draw_bboxes
                else []
            )

            if not blur_targets and not bbox_dets:
                # No visible change would result from saving a copy.
                result.skipped_no_change += 1
                continue

            source = _source_for(file)
            if source is None or not source.exists():
                result.skipped_missing_source += 1
                continue

            try:
                image = Image.open(source).convert("RGB")
            except OSError as e:
                result.errors.append(f"Could not open {source}: {e}")
                logger.exception(
                    f"annotated_copies: open failed for {source}"
                )
                continue

            # Blur first, so the box drawn over a blurred person sits
            # on the obscured pixels (matches legacy ordering).
            if blur_targets:
                radius = _blur_radius(image)
                for det in blur_targets:
                    _blur_region(image, det, radius)
                    result.blurred_box_count += 1

            if bbox_dets:
                rgba = image.convert("RGBA")
                overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay, "RGBA")
                # Sizes scale with this image's resolution.
                m = render_metrics(*rgba.size)
                font = _font(m.font)
                for det in bbox_dets:
                    _draw_one(draw, det, rgba.size, m, font, name_mode)
                    result.bbox_count += 1
                image = Image.alpha_composite(rgba, overlay).convert("RGB")

            destinations = _resolve_destinations(file, ctx, result)

            # Tag set is the same across destinations for one source.
            tag_set = build_tag_set(
                db,
                file,
                project,
                APP_VERSION,
                media_threshold=threshold,
                excluded_label_ids=excluded_label_ids,
            )

            for dest in destinations:
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    image.save(dest)
                except OSError as e:
                    result.errors.append(
                        f"Could not save {dest}: {e}"
                    )
                    logger.exception(
                        f"annotated_copies: save failed for {dest}"
                    )
                    continue

                result.written_count += 1

                # PIL drops EXIF on save, so re-write the predictions
                # tag set onto every saved file.
                if tag_set is not None and is_image_path(dest):
                    try:
                        exif_batch.write(dest, tag_set)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"annotated_copies: EXIF write failed "
                            f"for {dest}: {e}"
                        )

    logger.info(
        f"annotated_copies: project={project_id} "
        f"draw_bboxes={draw_bboxes} anonymise={anonymise} "
        f"written={result.written_count} "
        f"bboxes={result.bbox_count} "
        f"blurred={result.blurred_box_count} "
        f"no_change={result.skipped_no_change} "
        f"missing={result.skipped_missing_source} "
        f"excluded={result.skipped_excluded}"
    )
    return result
