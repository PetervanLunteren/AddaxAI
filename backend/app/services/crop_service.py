"""
Crop service - generates and caches detection crop thumbnails.

Crops the source image at the detection's bounding box and resizes
to a square thumbnail. Cached on disk to avoid re-cropping.
"""

from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import get_default_user_data_dir
from app.core.logging_config import get_logger
from app.models import Detection, File

logger = get_logger(__name__)

CACHE_DIR = get_default_user_data_dir() / "cache" / "crops"


def _resolve_image_path(file: File) -> Path | None:
    """Resolve the source image path for a file (image, video best_frame, or frame)."""
    if file.file_type == "video" and file.best_frame_path:
        p = Path(file.best_frame_path)
        if p.exists():
            return p
    if file.file_type == "frame" and file.file_path:
        p = Path(file.file_path)
        if p.exists():
            return p
    if file.file_path:
        p = Path(file.file_path)
        if p.exists():
            return p
    return None


def get_or_create_crop(detection_id: str, size: int, db: Session) -> Path | None:
    """
    Get or create a cropped thumbnail for a detection.

    Returns the path to the cached crop JPEG, or None if source image is missing.
    """
    cache_path = CACHE_DIR / f"{detection_id}_{size}.jpg"
    if cache_path.exists():
        return cache_path

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

        # Convert normalized bbox to pixel coords
        left = int(detection.bbox_x * w)
        top = int(detection.bbox_y * h)
        right = int((detection.bbox_x + detection.bbox_width) * w)
        bottom = int((detection.bbox_y + detection.bbox_height) * h)

        # Clamp to image bounds
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        if right <= left or bottom <= top:
            logger.warning(f"Invalid crop bbox for detection {detection_id}")
            return None

        crop = img.crop((left, top, right, bottom))
        crop = crop.resize((size, size), Image.LANCZOS)

        # Convert to RGB if needed (e.g., RGBA or palette images)
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        crop.save(cache_path, "JPEG", quality=85)
        return cache_path

    except Exception:
        logger.exception(f"Failed to create crop for detection {detection_id}")
        return None


def invalidate_crop_cache(detection_id: str) -> None:
    """Delete all cached crops for a detection (e.g., after bbox edit)."""
    if not CACHE_DIR.exists():
        return
    for cached in CACHE_DIR.glob(f"{detection_id}_*.jpg"):
        cached.unlink(missing_ok=True)
