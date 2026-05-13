"""
Crop service - generates and caches detection crop thumbnails.

Crops the source image at the detection's bounding box, expands to a
square with context padding, and resizes to a thumbnail. When the crop
extends beyond the image, the overflow is filled with a blurred edge
extension so the bbox stays centered. Cached in an in-memory LRU.
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
) -> tuple[int, int, int, int]:
    """Compute square crop region centered on bbox with padding.

    Returns (left, top, right, bottom) in pixel coords. Values may be
    negative or exceed image dimensions — the caller handles overflow
    with blurred edge fill.
    """
    bx, by = bbox_x * img_w, bbox_y * img_h
    bw, bh = bbox_w * img_w, bbox_h * img_h

    max_side = max(bw, bh)
    pad = max_side * padding
    crop_side = max_side + 2 * pad

    cx, cy = bx + bw / 2, by + bh / 2
    left = cx - crop_side / 2
    top = cy - crop_side / 2

    return int(left), int(top), int(left + crop_side), int(top + crop_side)


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


def _resolve_image_path(file: File) -> Path | None:
    """Resolve the source image path for a file.

    Images render from `file.file_path`. Videos render from
    `file.best_frame_path` (the canonical thumbnail written by the
    classifier worker or the no-classifier streaming pass). We never
    fall back to the .mp4 path: that would hand a video file to PIL,
    which crashes loudly downstream. Returning None lets the caller
    surface a clean "no thumbnail" state.
    """
    if file.file_type == "video":
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


def get_or_create_crop(detection_id: str, size: int, db: Session) -> bytes | None:
    """
    Get or create a cropped thumbnail for a detection.

    Returns JPEG bytes from an in-memory LRU cache, or None if the
    source image is missing.
    """
    cache_key = f"{detection_id}_{size}"

    if cache_key in _cache:
        _cache.move_to_end(cache_key)
        return _cache[cache_key]

    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        return None

    file = db.query(File).filter(File.id == detection.file_id).first()
    if not file:
        return None

    image_path = _resolve_image_path(file)
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
        )

        crop_w = right - left
        crop_h = bottom - top
        if crop_w <= 0 or crop_h <= 0:
            logger.warning(f"Invalid crop bbox for detection {detection_id}")
            return None

        crop = _crop_with_blur_fill(img, left, top, right, bottom)
        crop = crop.resize((size, size), Image.LANCZOS)

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
