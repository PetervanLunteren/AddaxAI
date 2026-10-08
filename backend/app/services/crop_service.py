"""
Crop service - generates and caches detection crop thumbnails.

Crops the source image at the detection's bounding box, expands to a
square with context padding, and resizes to a thumbnail. When the crop
extends beyond the image, the overflow is filled with a blurred edge
extension so the bbox stays centered. Cached in an in-memory LRU.

A caller that shows the crop in a non-square frame (the dashboard's hero
photo) asks for an `aspect`. The window then takes that shape and slides
inside the photo, so the rest of the picture fills the frame instead of
blur. Square crops never slide: the Labels grid draws its bbox overlay
from a centred square (`_compute_crop_bbox` in similarity_script).
"""

import io
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageFilter
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import Detection, File

logger = get_logger(__name__)

_MAX_CACHE_ENTRIES = 2000
_cache: OrderedDict[str, bytes] = OrderedDict()

_BLUR_RADIUS = 30


def compute_expanded_crop_region(
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    img_w: int,
    img_h: int,
    padding: float = 0.10,
    aspect: float | None = None,
) -> tuple[int, int, int, int]:
    """Compute the crop region around a bbox with padding.

    Without `aspect`: a square centered on the bbox. With `aspect`
    (width / height): the padded bbox widened to that shape, then slid
    inside the photo so the frame fills with real pixels, the way a
    camera would pan, rather than with blur. The whole padded bbox stays
    in view either way. Only a window larger than the photo itself is
    left centered, and its overflow blurred like a square crop's.

    Returns (left, top, right, bottom) in pixel coords. Values may be
    negative or exceed image dimensions — the caller handles overflow
    with blurred edge fill.
    """
    bx, by = bbox_x * img_w, bbox_y * img_h
    bw, bh = bbox_w * img_w, bbox_h * img_h

    max_side = max(bw, bh)
    pad = max_side * padding
    cx, cy = bx + bw / 2, by + bh / 2

    if aspect is None:
        crop_side = max_side + 2 * pad
        left = cx - crop_side / 2
        top = cy - crop_side / 2
        return int(left), int(top), int(left + crop_side), int(top + crop_side)

    win_w = max(bw + 2 * pad, (bh + 2 * pad) * aspect)
    win_h = win_w / aspect
    left = _slide_inside(cx - win_w / 2, win_w, img_w)
    top = _slide_inside(cy - win_h / 2, win_h, img_h)
    return int(left), int(top), int(left + win_w), int(top + win_h)


def _slide_inside(start: float, length: float, limit: int) -> float:
    """Move a window along one axis so it lies within [0, limit].

    A window longer than the photo cannot fit, so it keeps its centered
    start and the overflow is blurred. Sliding a window that contains the
    bbox toward the inside of the photo keeps the bbox in it.
    """
    if length >= limit:
        return start
    return min(max(start, 0.0), limit - length)


def _crop_with_blur_fill(
    img: Image.Image, left: int, top: int, right: int, bottom: int
) -> Image.Image:
    """Crop a region from the image, filling out-of-bounds areas with blurred edge."""
    img_w, img_h = img.size
    crop_w = right - left
    crop_h = bottom - top

    # Fast path: entirely within bounds
    if left >= 0 and top >= 0 and right <= img_w and bottom <= img_h:
        return img.crop((left, top, right, bottom))

    # Clamp to valid region
    valid_left = max(0, left)
    valid_top = max(0, top)
    valid_right = min(img_w, right)
    valid_bottom = min(img_h, bottom)

    if valid_right <= valid_left or valid_bottom <= valid_top:
        return img.crop((0, 0, min(crop_w, img_w), min(crop_h, img_h)))

    # Stretch the valid region to fill the full canvas, then blur heavily.
    # This gives the overflow areas natural image colors instead of
    # replicating edge pixels (which copies black info bars on camera traps).
    valid_crop = img.crop((valid_left, valid_top, valid_right, valid_bottom))
    canvas = valid_crop.resize((crop_w, crop_h), Image.BILINEAR)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=_BLUR_RADIUS))

    # Paste sharp original on top
    paste_x = valid_left - left
    paste_y = valid_top - top
    canvas.paste(valid_crop, (paste_x, paste_y))

    return canvas


def _resolve_image_path(file: File, detection: Detection) -> Path | None:
    """Resolve the source image to crop this detection from.

    Images render from `file.file_path`. Videos render from
    `file.best_frame_path` (the canonical thumbnail written by the
    classifier worker or the no-classifier streaming pass). We never
    fall back to the .mp4 path: that would hand a video file to PIL,
    which crashes loudly downstream. Returning None lets the caller
    surface a clean "no thumbnail" state.

    A video detection off the best frame gets None. It is the only frame
    on disk, so cropping it at a bbox from another frame produces a
    confident picture of the wrong place: the animal has moved, and the
    crop shows the leaf litter it left behind. That looked like a working
    thumbnail for as long as the subject sat still, which is why it went
    unnoticed. No image is the honest answer; the video player is where
    those detections are meant to be seen.
    """
    if file.file_type == "video":
        if detection.frame_number != file.best_frame_number:
            return None
        if file.best_frame_path:
            p = Path(file.best_frame_path)
            if p.exists():
                return p
        return None
    if file.file_path:
        p = Path(file.file_path)
        if p.exists():
            return p
    return None


def get_or_create_crop(
    detection_id: str, size: int, db: Session, aspect: float | None = None
) -> bytes | None:
    """
    Get or create a cropped thumbnail for a detection.

    `size` is the long side in pixels; `aspect` (width / height) asks for
    a shaped crop, see `compute_expanded_crop_region`. Returns JPEG bytes
    from an in-memory LRU cache, or None if the source image is missing.
    """
    cache_key = f"{detection_id}_{size}" + (f"_{aspect}" if aspect else "")

    if cache_key in _cache:
        _cache.move_to_end(cache_key)
        return _cache[cache_key]

    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        return None

    # Event-level observations have no bbox to crop. Caller (the crop
    # endpoint) will surface this as a 404; the UI never asks for these
    # because no-bbox rows render without a thumbnail.
    if detection.bbox_x is None:
        return None

    file = db.query(File).filter(File.id == detection.file_id).first()
    if not file:
        return None

    image_path = _resolve_image_path(file, detection)
    if not image_path:
        return None

    try:
        img = Image.open(image_path)
        w, h = img.size

        if img.mode != "RGB":
            img = img.convert("RGB")

        left, top, right, bottom = compute_expanded_crop_region(
            detection.bbox_x,
            detection.bbox_y,
            detection.bbox_width,
            detection.bbox_height,
            w,
            h,
            aspect=aspect,
        )

        crop_w = right - left
        crop_h = bottom - top
        if crop_w <= 0 or crop_h <= 0:
            logger.warning(f"Invalid crop bbox for detection {detection_id}")
            return None

        crop = _crop_with_blur_fill(img, left, top, right, bottom)
        if aspect is None or aspect == 1:
            out_size = (size, size)
        elif aspect > 1:
            out_size = (size, max(1, round(size / aspect)))
        else:
            out_size = (max(1, round(size * aspect)), size)
        crop = crop.resize(out_size, Image.LANCZOS)

        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

        # LRU eviction
        _cache[cache_key] = jpeg_bytes
        if len(_cache) > _MAX_CACHE_ENTRIES:
            _cache.popitem(last=False)

        return jpeg_bytes

    except Exception:
        logger.exception(f"Failed to create crop for detection {detection_id}")
        return None


def invalidate_crop_cache(detection_id: str) -> None:
    """Evict all cached crops for a detection (e.g., after bbox edit)."""
    keys_to_remove = [k for k in _cache if k.startswith(f"{detection_id}_")]
    for k in keys_to_remove:
        del _cache[k]
