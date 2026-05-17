"""Draw rounded boxes + pill labels on copies of media files.

Visual style matches the verify page (see `_visualisation_style.py`
for the shared spec): rounded-corner bbox outlines at 50% opacity,
rounded label pills with a coloured dot, white text on a dim
background, two-line layout when a species label is present. The
species colour comes from the same FNV-1a + RGB gradient algorithm
the frontend uses, so a label that's "teal-green" in the verify grid
is also "teal-green" on the saved JPEG.

For images: the source file is loaded, detections are drawn on top,
the result is saved into `target/<original_name>`.

For videos: there is no per-frame video output here. Instead we
visualise the file's best frame — the per-video representative JPEG
the pipeline already wrote to disk — into `target/<video_stem>.jpg`.
The best frame is the canonical visual stand-in for a video
everywhere else in the UI.

Copy semantics only. Source files are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app import __version__ as APP_VERSION
from app.core.logging_config import get_logger
from app.models import Deployment, Detection, File, Project

from ._exif_writer import ExifBatch, build_tag_set
from ._label_filter import file_is_dropped_by_filter
from ._visualisation_style import (
    BBOX_CORNER_RADIUS,
    BBOX_OPACITY,
    BBOX_STROKE_WIDTH,
    DOT_R,
    FONT_LG,
    FONT_SM,
    LINE_GAP,
    PILL_BG_RGBA,
    PILL_PAD_X,
    PILL_PAD_Y,
    TEXT_START_X,
    WHITE,
    WHITE_DIM,
    detection_color,
)

logger = get_logger(__name__)


@dataclass
class VisualisedImagesResult:
    """Summary of a visualised-images run.

    ``skipped_excluded`` counts animal files where every passing
    label was in the user's species exclusion set, so no
    visualisation was produced.
    """

    written_count: int = 0
    skipped_no_bbox: int = 0
    skipped_missing_source: int = 0
    skipped_excluded: int = 0
    renamed_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "written_count": self.written_count,
            "skipped_no_bbox": self.skipped_no_bbox,
            "skipped_missing_source": self.skipped_missing_source,
            "skipped_excluded": self.skipped_excluded,
            "renamed_count": self.renamed_count,
            "errors": list(self.errors),
        }


# ─────────────────────────────────────────────────────────────────
# Font handling
# ─────────────────────────────────────────────────────────────────
#
# Two sizes (small caption, larger label) just like the frontend pill.
# Pillow's `ImageFont.load_default(size=...)` returns the bundled
# Noto Sans TrueType in Pillow 10+, which gives readable text. We
# don't ship Arial explicitly because the visual difference is
# small enough that nailing the exact frontend font is not worth
# bundling another asset.


def _load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    """Return (small, large) fonts for the pill layout.

    Pillow's bundled default truetype scales reasonably; we use the
    same point sizes the frontend pill uses so visual proportions
    match.
    """
    try:
        small = ImageFont.load_default(size=FONT_SM)
        large = ImageFont.load_default(size=FONT_LG)
    except (TypeError, AttributeError):  # Pillow < 10 fallback
        default = ImageFont.load_default()
        small = default
        large = default
    return small, large


# ─────────────────────────────────────────────────────────────────
# Text metrics
# ─────────────────────────────────────────────────────────────────


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    """Pillow-version-agnostic text width."""
    if hasattr(draw, "textbbox"):
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        return right - left
    return draw.textsize(text, font=font)[0]  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────
# Pill layout
# ─────────────────────────────────────────────────────────────────


@dataclass
class _PillLayout:
    category_text: str
    label_text: str
    has_label: bool
    pill_width: int
    pill_height: int
    color: tuple[int, int, int]


def _compute_pill_layout(
    draw: ImageDraw.ImageDraw,
    detection: Detection,
    font_sm,
    font_lg,
) -> _PillLayout:
    """Build the pill geometry for a single detection.

    Mirrors `computePillLayout` in detection-overlay.ts: a two-line
    pill ("Category XX%" / "Label YY%") when a species label is
    present, otherwise a single-line pill with just the category and
    confidence. Width is text-width + padding, so each pill is as
    wide as it needs to be and no wider.
    """
    color = detection_color(detection.label, detection.category)
    has_label = bool(detection.label)

    category_text = (
        f"{detection.category.capitalize()} "
        f"{int(round(detection.confidence * 100))}%"
    )

    if has_label:
        display_name = detection.display_name or detection.label
        label_text = (
            f"{display_name[:1].upper()}{display_name[1:]} "
            f"{int(round((detection.label_confidence or detection.confidence) * 100))}%"
        )
        pill_height = PILL_PAD_Y + FONT_SM + LINE_GAP + FONT_LG + PILL_PAD_Y
        w1 = _text_width(draw, category_text, font_sm)
        w2 = _text_width(draw, label_text, font_lg)
        pill_width = TEXT_START_X + max(w1, w2) + PILL_PAD_X
    else:
        label_text = ""
        pill_height = PILL_PAD_Y + FONT_LG + PILL_PAD_Y
        tw = _text_width(draw, category_text, font_lg)
        pill_width = TEXT_START_X + tw + PILL_PAD_X

    return _PillLayout(
        category_text=category_text,
        label_text=label_text,
        has_label=has_label,
        pill_width=pill_width,
        pill_height=pill_height,
        color=color,
    )


# ─────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────


def _draw_one(
    overlay: Image.Image,
    draw: ImageDraw.ImageDraw,
    detection: Detection,
    image_size: tuple[int, int],
    font_sm,
    font_lg,
) -> None:
    """Draw a single bbox + its pill label onto the overlay in place.

    The overlay is RGBA; the bbox stroke uses the detection colour at
    `BBOX_OPACITY`, the pill background uses `PILL_BG_RGBA` (a dim
    black). Composite of overlay onto the source image gives the
    final translucent look.
    """
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
    stroke_rgba = (color[0], color[1], color[2], int(round(BBOX_OPACITY * 255)))

    # Box outline — rounded corners.
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=BBOX_CORNER_RADIUS,
        outline=stroke_rgba,
        width=BBOX_STROKE_WIDTH,
    )

    # Pill layout
    pill = _compute_pill_layout(draw, detection, font_sm, font_lg)

    # Clamp pill horizontally so it stays on the canvas, and place
    # above the box; if there isn't room above, drop it just inside
    # the top of the box. Same fallback as AnnotationCanvas.
    pill_x = max(0, min(x0, img_w - pill.pill_width))
    pill_y = y0 - pill.pill_height if y0 - pill.pill_height >= 0 else y0

    # Pill background — rounded rectangle with dim black fill.
    draw.rounded_rectangle(
        (pill_x, pill_y, pill_x + pill.pill_width, pill_y + pill.pill_height),
        radius=BBOX_CORNER_RADIUS,
        fill=PILL_BG_RGBA,
    )

    # Colored species/category dot at the left of the pill.
    dot_cx = pill_x + PILL_PAD_X + DOT_R
    dot_cy = pill_y + pill.pill_height // 2
    draw.ellipse(
        (dot_cx - DOT_R, dot_cy - DOT_R, dot_cx + DOT_R, dot_cy + DOT_R),
        fill=(*color, 255),
    )

    text_x = pill_x + TEXT_START_X
    if pill.has_label:
        # Small caption (category + confidence) on top, bold label
        # text below. The frontend pill uses bold for the lower line;
        # Pillow's default font doesn't have a bold variant we can
        # rely on, so we render both lines plain — the size and dim
        # category line still produce the same visual hierarchy.
        draw.text(
            (text_x, pill_y + PILL_PAD_Y),
            pill.category_text,
            fill=WHITE_DIM,
            font=font_sm,
        )
        draw.text(
            (text_x, pill_y + PILL_PAD_Y + FONT_SM + LINE_GAP),
            pill.label_text,
            fill=(*WHITE, 255),
            font=font_lg,
        )
    else:
        draw.text(
            (text_x, pill_y + PILL_PAD_Y),
            pill.category_text,
            fill=(*WHITE, 255),
            font=font_lg,
        )

    # Acknowledge `overlay` is part of the API surface even though
    # we drew through `draw`; keeping the parameter makes the helper
    # symmetrical and reads better at the call site.
    _ = overlay


# ─────────────────────────────────────────────────────────────────
# Source resolution + detection lookup
# ─────────────────────────────────────────────────────────────────


def _detections_to_draw(
    db: Session,
    file: File,
    threshold: float,
    excluded_label_ids: frozenset[str] | None = None,
) -> list[Detection]:
    """Detections to draw for a single file.

    For images: every detection on the file that passes threshold (or
    is verified) and has a bbox.

    For videos: only the detections anchored to the best frame, since
    that is the frame we are about to draw on.

    Detections whose label is in ``excluded_label_ids`` are filtered
    out so the rendered copy does not show boxes for species the
    user excluded.
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
        stmt = stmt.where(Detection.frame_number == file.best_frame_number)
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


def _source_for(file: File) -> Path | None:
    """Resolve the on-disk source to draw on.

    Images use their own path. Videos use the pipeline-written best
    frame JPEG. Returns None when neither is available.
    """
    if file.file_type == "image":
        return Path(file.file_path)
    if file.file_type == "video":
        if not file.best_frame_path:
            return None
        return Path(file.best_frame_path)
    return None


def _destination_name(file: File) -> str:
    """Image keeps its name; video uses stem.jpg (still frame)."""
    source = Path(file.file_path)
    if file.file_type == "video":
        return source.stem + ".jpg"
    return source.name


def _unique_destination(target_dir: Path, source_name: str) -> tuple[Path, bool]:
    """Collision-safe destination path."""
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


def visualise_images(
    db: Session,
    project_id: str,
    target_dir: Path,
    *,
    excluded_label_ids: frozenset[str] | None = None,
) -> VisualisedImagesResult:
    """Write a visualised copy of every file in the project.

    Animal files whose every passing label is in
    ``excluded_label_ids`` are skipped entirely. Detection boxes for
    excluded species are filtered from the visualisation so a
    surviving file shows only the included detections.
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

    result = VisualisedImagesResult()
    font_sm, font_lg = _load_fonts()

    with ExifBatch() as exif_batch:
        for file in files:
            source = _source_for(file)
            if source is None or not source.exists():
                result.skipped_missing_source += 1
                continue

            if file_is_dropped_by_filter(
                db, file, threshold, excluded_label_ids
            ):
                result.skipped_excluded += 1
                continue

            detections = _detections_to_draw(
                db, file, threshold, excluded_label_ids
            )
            if not detections:
                # Visualisation only makes sense when there's something
                # to draw. Skipping keeps blank folders clean.
                result.skipped_no_bbox += 1
                continue

            try:
                base = Image.open(source).convert("RGBA")
            except OSError as e:
                result.errors.append(f"Could not open {source}: {e}")
                logger.exception(
                    f"visualised_images: open failed for {source}"
                )
                continue

            # Draw on a transparent overlay so per-shape alpha (50%
            # box stroke, 50% pill background) composites correctly.
            # Pillow's ImageDraw on an RGB image cannot apply
            # per-pixel alpha; an RGBA overlay merged onto the base
            # does.
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay, "RGBA")
            for det in detections:
                _draw_one(overlay, draw, det, base.size, font_sm, font_lg)

            composite = Image.alpha_composite(base, overlay).convert("RGB")

            destination, renamed = _unique_destination(
                target_dir, _destination_name(file)
            )
            try:
                composite.save(destination)
            except OSError as e:
                result.errors.append(f"Could not save {destination}: {e}")
                logger.exception(
                    f"visualised_images: save failed for {destination}"
                )
                continue

            result.written_count += 1
            if renamed:
                result.renamed_count += 1

            # Silent EXIF write on the visualised JPEG so the labels
            # travel with the file when shared.
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
                        f"visualised_images: EXIF write failed for "
                        f"{destination}: {e}"
                    )

    logger.info(
        f"visualised_images: project={project_id} "
        f"written={result.written_count} "
        f"no_bbox={result.skipped_no_bbox} "
        f"missing={result.skipped_missing_source} "
        f"excluded={result.skipped_excluded}"
    )
    return result
