"""Image utility functions for AddaxAI.

Corruption detection, EXIF timestamp extraction, burst/series grouping,
and bounding-box blurring. Requires Pillow; OpenCV only for blur_box.
"""

import datetime
import os
import re

import PIL.ExifTags
from PIL import Image, ImageFile


# ---------------------------------------------------------------------------
#  Image corruption
# ---------------------------------------------------------------------------

def is_image_corrupted(image_path):
    """Check if an image file is corrupted by attempting to fully load it.

    Returns:
        False if the image loads successfully, True if it fails.
    """
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(image_path) as img:
            img.load()
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return False
    except Exception:
        return True


def check_images(image_list_file):
    """Read a file of image paths and return a list of corrupted ones.

    Args:
        image_list_file: Text file with one image path per line.

    Returns:
        List of paths to corrupted images.
    """
    corrupted_images = []
    with open(image_list_file, 'r') as f:
        image_paths = f.read().splitlines()
    for image_path in image_paths:
        if os.path.exists(image_path):
            if is_image_corrupted(image_path):
                corrupted_images.append(image_path)
    return corrupted_images


def fix_images(image_paths):
    """Attempt to fix truncated images by re-opening and re-saving them.

    Preserves EXIF data when possible. Prints a warning for images that
    cannot be fixed.
    """
    for image_path in image_paths:
        if os.path.exists(image_path):
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                with Image.open(image_path) as img:
                    img_copy = img.copy()
                    img_copy.save(image_path, format=img.format,
                                 exif=img.info.get('exif'))
            except Exception as e:
                print(f"Could not fix image: {e}")


# ---------------------------------------------------------------------------
#  Timestamp extraction
# ---------------------------------------------------------------------------

def _parse_timestamp_from_filename(filename):
    """Try common camera filename timestamp patterns.

    Patterns tried: YYYYMMDDHHMMSS, YYYYMMDD_HHMMSS, YYYY-MM-DD_HHMMSS.

    Returns:
        datetime.datetime or None.
    """
    patterns = [
        r'(?P<ts>\d{14})',
        r'(?P<ts>\d{8}[_-]\d{6})',
        r'(?P<ts>\d{4}-\d{2}-\d{2}[_-]\d{6})',
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if not m:
            continue
        ts = re.sub(r'[_-]', '', m.group('ts'))
        try:
            return datetime.datetime.strptime(ts, '%Y%m%d%H%M%S')
        except Exception:
            continue
    return None


def get_image_timestamp(src_dir, rel_path):
    """Return a datetime for the given image file.

    Uses a 3-level fallback:
      1. EXIF DateTimeOriginal / DateTime
      2. Filename timestamp patterns
      3. Filesystem modification time

    Args:
        src_dir: Base directory containing the image.
        rel_path: Relative path from src_dir to the image.

    Returns:
        datetime.datetime or None.
    """
    abs_path = os.path.join(src_dir, rel_path)

    # (1) EXIF
    try:
        img = Image.open(abs_path)
        exif = img.getexif()
        if exif:
            tag_map = {v: k for k, v in PIL.ExifTags.TAGS.items()}
            for tag_name in ("DateTimeOriginal", "DateTime"):
                if tag_name in tag_map:
                    tid = tag_map[tag_name]
                    if tid in exif:
                        val = exif.get(tid)
                        if isinstance(val, bytes):
                            val = val.decode(errors="ignore")
                        if val:
                            val = val.replace('\x00', '').strip()
                            try:
                                return datetime.datetime.strptime(
                                    val, "%Y:%m:%d %H:%M:%S")
                            except Exception:
                                try:
                                    return datetime.datetime.strptime(
                                        val, "%Y-%m-%d %H:%M:%S")
                                except Exception:
                                    pass
    except Exception:
        pass

    # (2) Filename
    try:
        ts = _parse_timestamp_from_filename(os.path.basename(rel_path))
        if ts:
            return ts
    except Exception:
        pass

    # (3) Filesystem mtime
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(abs_path))
    except Exception:
        return None


def build_image_timestamp_index(src_dir, file_list):
    """Build a dict mapping relative paths to their timestamps.

    Args:
        src_dir: Base directory containing the images.
        file_list: List of relative paths to index.

    Returns:
        Dict of {rel_path: datetime or None}.
    """
    idx = {}
    for rel in file_list:
        try:
            idx[rel] = get_image_timestamp(src_dir, rel)
        except Exception:
            idx[rel] = None
    return idx


# ---------------------------------------------------------------------------
#  Burst / series detection
# ---------------------------------------------------------------------------

def _camera_prefix_of_filename(filename):
    """Extract the camera identifier prefix from a filename.

    Returns the part before the first 14-digit or 8+6-digit timestamp
    pattern, or None if no timestamp is found.
    """
    m = re.search(r'\d{14}', filename)
    if m:
        return filename[:m.start()]
    m = re.search(r'\d{8}[_-]\d{6}', filename)
    if m:
        return filename[:m.start()]
    return None


def find_series_images(target_rel_path, timestamp_index, window_seconds=10,
                       require_same_camera=True):
    """Find images in the same burst as the target image.

    Args:
        target_rel_path: Relative path of the target image.
        timestamp_index: Dict from build_image_timestamp_index.
        window_seconds: Max time gap to consider as same burst.
        require_same_camera: Only include files with the same filename prefix.

    Returns:
        List of relative paths in chronological order.
    """
    if target_rel_path not in timestamp_index:
        return [target_rel_path]
    t0 = timestamp_index.get(target_rel_path)
    if t0 is None:
        return [target_rel_path]

    prefix0 = (_camera_prefix_of_filename(os.path.basename(target_rel_path))
               if require_same_camera else None)
    candidates = []
    for rel, t in timestamp_index.items():
        if t is None:
            continue
        if require_same_camera:
            prefix = _camera_prefix_of_filename(os.path.basename(rel))
            if (prefix0 is not None and prefix is not None
                    and prefix0 != prefix):
                continue
        if abs((t - t0).total_seconds()) <= window_seconds:
            candidates.append((rel, t))
    candidates.sort(key=lambda x: x[1] or datetime.datetime.max)
    return [c[0] for c in candidates]


# ---------------------------------------------------------------------------
#  Bounding-box blur (requires OpenCV)
# ---------------------------------------------------------------------------

def blur_box(image, bbox_left, bbox_top, bbox_right, bbox_bottom,
             image_width, image_height):
    """Blur a rectangular region in an OpenCV image (numpy array).

    Used for privacy protection (e.g., blurring detected people).

    Args:
        image: numpy array (H, W, C) from cv2.imread.
        bbox_left, bbox_top, bbox_right, bbox_bottom: Bounding box coords.
        image_width, image_height: Image dimensions for validation.

    Returns:
        The modified image array with the region blurred.

    Raises:
        ValueError: If bounding box coordinates are invalid.
    """
    import cv2

    x1, y1, x2, y2 = map(int, [bbox_left, bbox_top, bbox_right, bbox_bottom])
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        raise ValueError(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Extracted ROI is empty. Check the bounding box coordinates.")
    blurred_roi = cv2.GaussianBlur(roi, (71, 71), 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image
