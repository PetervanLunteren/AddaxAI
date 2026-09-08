"""
Crop service - generates and caches detection crop thumbnails.

Crops the source image at the detection's bounding box, expands to a
square with context padding, and resizes to a thumbnail. When the crop
extends beyond the image, the overflow is filled with a blurred edge
extension so the bbox stays centered. The geometry lives in
`app/ml/inference/crop_box.py`, shared with the tracking script, which
cuts a track's card the same way while it decodes the video. Cached in
an in-memory LRU.
"""

import io
from collections import OrderedDict
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.embedding_utils import video_pixels_for
from app.ml.inference.crop_box import compute_expanded_crop_region, crop_with_blur_fill
from app.models import Detection, File
from app.services.video_frame_service import decode_frame_jpeg

logger = get_logger(__name__)

_MAX_CACHE_ENTRIES = 2000
_cache: OrderedDict[str, bytes] = OrderedDict()


_WHOLE_IMAGE = [0.0, 0.0, 1.0, 1.0]


def _resolve_source(
    file: File, detection: Detection
) -> tuple[Path | bytes, list[float]] | None:
    """The picture to cut this detection's card from, and the box in it.

    An image is cropped at the box. A video box is a card only on its
    track's representative frame; there the picture is the track's
    stored crop (the whole file), or the cover frame when the card has
    no crop and sits on it, or, for a card on any other frame (a drawn
    box, a legacy verified box), that frame decoded on request. Never
    the video container itself: PIL cannot open it. A box that is not a
    card gets None, which the caller answers as "no thumbnail": cropping
    the cover at a box from another moment gave a confident picture of
    the wrong place.
    """
    if file.file_type != "video":
        if file.file_path and Path(file.file_path).exists():
            return Path(file.file_path), [
                detection.bbox_x, detection.bbox_y,
                detection.bbox_width, detection.bbox_height,
            ]
        return None
    track = detection.track
    if (
        track is None
        or detection.frame_number is None
        or detection.frame_number != track.representative_frame_number
    ):
        return None
    pixels = video_pixels_for(detection, file)
    if pixels is not None:
        path, bbox = pixels
        return (Path(path), bbox) if Path(path).exists() else None
    if not file.file_path:
        return None
    jpeg = decode_frame_jpeg(file.file_path, detection.frame_number, file.frame_rate)
    if jpeg is None:
        return None
    return jpeg, [
        detection.bbox_x, detection.bbox_y, detection.bbox_width, detection.bbox_height,
    ]


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

    # Event-level observations have no bbox to crop. Caller (the crop
    # endpoint) will surface this as a 404; the UI never asks for these
    # because no-bbox rows render without a thumbnail.
    if detection.bbox_x is None:
        return None

    file = db.query(File).filter(File.id == detection.file_id).first()
    if not file:
        return None

    source = _resolve_source(file, detection)
    if source is None:
        return None
    picture, bbox = source

    try:
        img = Image.open(picture if isinstance(picture, Path) else io.BytesIO(picture))
        w, h = img.size

        if img.mode != "RGB":
            img = img.convert("RGB")

        if bbox == _WHOLE_IMAGE:
            # A stored track crop is already the padded square.
            crop = img
        else:
            left, top, right, bottom = compute_expanded_crop_region(*bbox, w, h)
            crop_w = right - left
            crop_h = bottom - top
            if crop_w <= 0 or crop_h <= 0:
                logger.warning(f"Invalid crop bbox for detection {detection_id}")
                return None
            crop = crop_with_blur_fill(img, left, top, right, bottom)
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
