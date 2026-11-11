"""
Crop service - generates and caches detection crop thumbnails.

Crops the source image at the detection's bounding box, expands to a
square with context padding, and resizes to a thumbnail. When the crop
extends beyond the image, the overflow is filled with a blurred edge
extension so the bbox stays centered. The geometry lives in
`app/ml/inference/crop_box.py`, shared with the tracking script, which
cuts a track's card the same way while it decodes the video. Cached in
an in-memory LRU.

A caller that shows the crop in a non-square frame (the dashboard's hero
photo, the calibration dialog) asks for an `aspect`. The window then
takes that shape and slides inside the photo, so the rest of the picture
fills the frame instead of blur. Square crops never slide: the Labels
grid draws its bbox overlay from a centred square (`_compute_crop_bbox`
in similarity_script). A shaped crop is always cut from the full picture,
a video card's from its decoded frame: the track's stored crop is the
square already, and a shape cut out of that loses the animal's head and
feet (until 2026-11-11).
"""

import io
from collections import OrderedDict
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.ml.embedding_utils import video_pixels_for
from app.ml.inference.crop_box import compute_expanded_crop_region, crop_with_blur_fill
from app.ml.inference.image_io import open_image_rgb
from app.models import Detection, File
from app.services.video_frame_service import decode_frame_jpeg, warm_track_frames

logger = get_logger(__name__)

_MAX_CACHE_ENTRIES = 2000
_cache: OrderedDict[str, bytes] = OrderedDict()



def _resolve_source(
    file: File,
    detection: Detection,
    track_frames: list[int] | None = None,
    *,
    square: bool = True,
) -> tuple[Path | bytes, list[float] | None] | None:
    """The picture to cut this detection's card from, and the box in it.
    No box (None) says the picture is the card already, to be served as
    it is: a track's stored crop. Said outright rather than read off a
    box that spans the picture, because a real box can: a whole-image
    classifier stamps one on every photo, and until 2026-11-11 those
    photos were taken for stored crops and squeezed into the square.

    An image is cropped at the box. For a video the ladder is, in order:

    1. no track, or no frame number: None. Those rows are visible
       nowhere (`ml/detection_visibility.py`), like the sub-threshold
       noise, and a thumbnail would be the only place they surfaced.
    2. the track's representative frame, its card: the track's stored
       crop (already the padded square, so the whole file), else the
       cover frame when the card sits on it, else that frame decoded.
       A picture that is not on disk is passed over, so a card whose
       crop did not travel (a bundle without photos, connected to a
       copy of the videos that was never analysed) is decoded instead
       of left blank. So is the stored crop when the request is not
       ``square``: a shape needs the frame around the box.
    3. any other frame of the track: that frame decoded on request, cut
       at this box. Only the opened track on the Detections tab asks
       for these, and it asks for a screenful at once, so the chunk of
       the track's frames from this one on is decoded with one seek
       (`warm_track_frames`, given ``track_frames``) and the frame is
       then read from the cache.

    Rung 3 must never go through `video_pixels_for`, which returns the
    track's stored crop whenever there is one: every frame of an animal
    would then render the same picture, which looks entirely plausible
    and is wrong. It is the same failure as cropping the cover at a box
    from another moment, where 31 of 32 tiles of a walking person were
    the leaf litter they had already left.

    Never the video container itself: PIL cannot open it.
    """
    if file.file_type != "video":
        if file.file_path and Path(file.file_path).exists():
            return Path(file.file_path), [
                detection.bbox_x, detection.bbox_y,
                detection.bbox_width, detection.bbox_height,
            ]
        return None
    track = detection.track
    if track is None or detection.frame_number is None:
        return None
    own_bbox = [
        detection.bbox_x, detection.bbox_y, detection.bbox_width, detection.bbox_height,
    ]
    if detection.frame_number == track.representative_frame_number:
        pixels = video_pixels_for(detection, file)
        if pixels is not None:
            path, bbox = pixels
            stored_crop = path == track.crop_path
            if Path(path).exists() and (square or not stored_crop):
                return Path(path), None if stored_crop else bbox
    if not file.file_path:
        return None
    if track_frames:
        warm_track_frames(
            file.file_path, track_frames, detection.frame_number, file.frame_rate
        )
    jpeg = decode_frame_jpeg(file.file_path, detection.frame_number, file.frame_rate)
    if jpeg is None:
        return None
    return jpeg, own_bbox


def _track_frames_of(db: Session, file: File, detection: Detection) -> list[int] | None:
    """The frames of this detection's track, ascending, for the chunk
    decode of rung 3; None for a photo or an untracked box."""
    if file.file_type != "video" or detection.track_id is None:
        return None
    rows = (
        db.query(Detection.frame_number)
        .filter(Detection.track_id == detection.track_id, Detection.frame_number.isnot(None))
        .distinct()
        .order_by(Detection.frame_number)
        .all()
    )
    return [r[0] for r in rows]


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

    square = aspect is None or aspect == 1
    source = _resolve_source(
        file, detection, _track_frames_of(db, file, detection), square=square
    )
    if source is None:
        return None
    picture, bbox = source

    try:
        if isinstance(picture, Path):
            # Rotated per EXIF and RGB, as MegaDetector saw the photo: the
            # bbox fractions refer to that frame (see image_io).
            img = open_image_rgb(picture)
        else:
            # A frame decoded from the clip: no EXIF to honour.
            img = Image.open(io.BytesIO(picture)).convert("RGB")
        w, h = img.size

        if bbox is None:
            # A stored track crop is already the padded square, and only
            # a square request is handed one (`_resolve_source`).
            crop = img
        else:
            left, top, right, bottom = compute_expanded_crop_region(
                *bbox, w, h, aspect=aspect
            )
            crop_w = right - left
            crop_h = bottom - top
            if crop_w <= 0 or crop_h <= 0:
                logger.warning(f"Invalid crop bbox for detection {detection_id}")
                return None
            crop = crop_with_blur_fill(img, left, top, right, bottom)
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
