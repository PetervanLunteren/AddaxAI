"""
Folder scanner service for deployment preview.

Provides lightweight folder analysis before running MegaDetector:
- Count images and videos
- Sample files to check GPS coordinates
- Suggest site matching based on location

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- No silent failures
"""

import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from PIL import Image
from PIL.ExifTags import GPSTAGS

from app.core.logging_config import get_logger
from app.core.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.utils.media_dates import _DATE_FIELDS as _VIDEO_DATE_FIELDS
from app.utils.media_dates import extract_video_dates as _shared_extract_video_dates

logger = get_logger(__name__)

# Marker file dropped inside AddaxAI output folders. Any directory that
# contains it is a results folder and is skipped during scans, so the
# copies / visualisations it holds never get re-ingested as input media
# (this is what lets the save step default to a subfolder of the source).
OUTPUT_DIR_MARKER = ".addaxai-output"


def prune_unscannable_dirs(root: str, dirnames: list[str]) -> list[str]:
    """Filter an ``os.walk`` dir list down to the ones worth descending into.

    Drops dot-folders (``.addaxai`` etc.) and AddaxAI output folders (those
    carrying ``OUTPUT_DIR_MARKER``), so a previous run's separated /
    visualised copies are never re-ingested as input media. Shared by the
    preview scan here and the worker's input enumeration
    (``detection_worker.scan_folder_for_*``) so the two cannot drift —
    that drift is what let output folders get reprocessed.
    """
    return [
        d
        for d in dirnames
        if not d.startswith(".")
        and not os.path.exists(os.path.join(root, d, OUTPUT_DIR_MARKER))
    ]


class GPSCoordinates(TypedDict):
    """GPS coordinates extracted from EXIF."""

    latitude: float
    longitude: float


class SampleFilePreview(TypedDict):
    """A sample file with its extracted datetime."""

    path: str  # Relative to deployment folder
    file_datetime: str | None  # ISO format datetime, or None if unavailable


class FolderPreview(TypedDict):
    """Preview of a deployment folder."""

    image_count: int
    video_count: int
    total_count: int
    gps_location: GPSCoordinates | None
    sample_files: list[SampleFilePreview]
    start_date: str | None  # ISO format datetime
    end_date: str | None  # ISO format datetime
    missing_datetime: bool  # True if no EXIF dates found
    datetime_validation_log: list[str]  # Log of what was tried and why rejected


def scan_folder(folder_path: str, gps_sample_size: int = 10) -> FolderPreview:
    """
    Scan a deployment folder for preview information.

    Args:
        folder_path: Absolute path to deployment folder
        gps_sample_size: Number of random images to check for GPS

    Returns:
        FolderPreview with counts, GPS location if found, and sample files

    Raises:
        FileNotFoundError: If folder doesn't exist
        PermissionError: If folder isn't readable
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    # Recursively find all media files
    image_files: list[Path] = []
    video_files: list[Path] = []

    for root, dirs, files in os.walk(folder):
        # Skip dot-folders (.addaxai etc.) and any folder carrying the
        # output marker — that's an AddaxAI results folder whose copies
        # must not be scanned back in as input media.
        dirs[:] = prune_unscannable_dirs(root, dirs)
        for filename in files:
            file_path = Path(root) / filename
            ext = file_path.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(file_path)
            elif ext in VIDEO_EXTENSIONS:
                video_files.append(file_path)

    # Build the full file list for the "Adjust dates" modal. Users
    # browse files to compare burned-in pixel dates with extracted
    # datetimes. Both images and videos are included (camera trap
    # videos also have burned-in timestamps). The preview-image
    # endpoint handles videos by extracting the first frame via ffmpeg.
    #
    # Datetimes are NOT extracted here (10k files × ~3ms = 30 seconds
    # is too slow). The frontend fetches on demand via file-datetime.
    #
    # Sorted by filename (chronological for camera traps).
    all_media = sorted(image_files + video_files, key=lambda p: p.name)
    sample_files = [
        {"path": str(f.relative_to(folder)), "file_datetime": None}
        for f in all_media
    ]

    # Try to extract GPS from random sample of images
    gps_location = _extract_gps_from_sample(folder, image_files, gps_sample_size)

    # Extract date range from images and videos with validation
    start_date, end_date, validation_log = _extract_date_range(image_files, video_files)

    return FolderPreview(
        image_count=len(image_files),
        video_count=len(video_files),
        total_count=len(image_files) + len(video_files),
        gps_location=gps_location,
        sample_files=sample_files,
        start_date=start_date.isoformat() if start_date else None,
        end_date=end_date.isoformat() if end_date else None,
        missing_datetime=start_date is None or end_date is None,
        datetime_validation_log=validation_log,
    )


def _extract_gps_from_sample(
    folder: Path, image_files: list[Path], sample_size: int
) -> GPSCoordinates | None:
    """
    Extract GPS coordinates from a random sample of images.

    Checks up to 50 random images. Stops early after finding GPS in 5 images,
    then averages the coordinates.

    Returns:
        Average GPS coordinates or None if not found
    """
    if not image_files:
        return None

    # Sample up to 50 images
    max_sample = 50
    sample = random.sample(image_files, min(max_sample, len(image_files)))

    gps_coords: list[GPSCoordinates] = []

    for img_path in sample:
        try:
            coords = _extract_gps_from_image(img_path)
            if coords:
                gps_coords.append(coords)
        except Exception as e:
            # Skip files with corrupt EXIF or other issues, but log it
            logger.warning(f"Failed to extract GPS from {img_path.name}: {type(e).__name__}: {e}")
            continue

        # Stop early after finding GPS in 5 images
        if len(gps_coords) >= 5:
            break

    if not gps_coords:
        return None

    # Average the coordinates
    avg_lat = sum(c["latitude"] for c in gps_coords) / len(gps_coords)
    avg_lon = sum(c["longitude"] for c in gps_coords) / len(gps_coords)

    return GPSCoordinates(latitude=avg_lat, longitude=avg_lon)


def _extract_gps_from_image(img_path: Path) -> GPSCoordinates | None:
    """
    Extract GPS coordinates from a single image's EXIF data.

    Returns:
        GPS coordinates or None if not found
    """
    try:
        with Image.open(img_path) as img:
            exif_data = img.getexif()
            if not exif_data:
                return None

            # Get GPS IFD using get_ifd() method (0x8825 is GPS IFD tag)
            try:
                gps_ifd = exif_data.get_ifd(0x8825)
            except KeyError:
                # No GPS data in this image
                return None

            if not gps_ifd:
                return None

            # Parse GPS data using GPSTAGS
            gps_data = {}
            for tag_id, value in gps_ifd.items():
                tag_name = GPSTAGS.get(tag_id, tag_id)
                gps_data[tag_name] = value

            # Convert to decimal degrees
            lat = _convert_to_degrees(gps_data.get("GPSLatitude"), gps_data.get("GPSLatitudeRef"))
            lon = _convert_to_degrees(gps_data.get("GPSLongitude"), gps_data.get("GPSLongitudeRef"))

            if lat is not None and lon is not None:
                return GPSCoordinates(latitude=lat, longitude=lon)

            return None

    except Exception as e:
        # Image corrupt, not readable, or other error - log it
        logger.debug(f"Cannot read image {img_path.name}: {type(e).__name__}: {e}")
        return None


def _convert_to_degrees(
    coord_tuple: tuple[float, float, float] | None, ref: str | None
) -> float | None:
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal degrees.

    Args:
        coord_tuple: (degrees, minutes, seconds)
        ref: Reference ('N', 'S', 'E', 'W')

    Returns:
        Decimal degrees or None if invalid
    """
    if not coord_tuple or not ref:
        return None

    try:
        degrees, minutes, seconds = coord_tuple
        decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600

        # Apply sign based on reference
        if ref in ("S", "W"):
            decimal = -decimal

        return decimal
    except (ValueError, TypeError):
        return None


def _extract_date_range(
    image_files: list[Path],
    video_files: list[Path],
) -> tuple[datetime | None, datetime | None, list[str]]:
    """
    Extract date range from image and video EXIF datetime metadata with validation.

    For images, tries EXIF date tags in order of preference (the first two
    read from the Exif sub-IFD where they actually live):
    1. DateTimeOriginal (36867) - camera capture time
    2. DateTimeDigitized (36868) - when digitized
    3. DateTime (306) - file modification time in camera

    For videos, tries the exiftool fields listed in
    app.utils.media_dates._DATE_FIELDS (CreateDate first, then EXIF / MP4 /
    RIFF fallbacks).

    Samples the first 5 and last 5 files (sorted by filename) plus 100
    random files across the folder. Camera traps use sequential filenames
    (IMG_0001.jpg, VID_0001.mp4, etc.) so first / last give the extremes for
    a clean single-camera folder; the random draw covers stripped-EXIF
    extremes and multi-subfolder folders. The resulting range is a rough,
    sample-based estimate, surfaced date-only in the UI.

    Returns:
        Tuple of (start_date, end_date, validation_log)
        validation_log contains human-readable messages about what was tried
    """
    if not image_files and not video_files:
        return None, None, ["No image or video files found"]

    validation_log: list[str] = []

    # Sort by filename and sample first/last
    # Camera traps use sequential filenames, so this gives us chronological order
    sorted_images = sorted(image_files, key=lambda p: p.name)
    sorted_videos = sorted(video_files, key=lambda p: p.name)

    # Take first 5 and last 5 (or whatever is available)
    num_to_sample = 5

    # Sample images
    if len(sorted_images) <= num_to_sample * 2:
        image_sample = sorted_images
        validation_log.append(f"Checking EXIF metadata in all {len(image_sample)} images...")
    else:
        first_n = sorted_images[:num_to_sample]
        last_n = sorted_images[-num_to_sample:]
        image_sample = first_n + last_n
        validation_log.append(
            f"Checking EXIF metadata in {len(image_sample)} images "
            f"(first {num_to_sample} and last {num_to_sample} sorted by filename) "
            f"out of {len(image_files)} total..."
        )

    # Sample videos
    if len(sorted_videos) <= num_to_sample * 2:
        video_sample = sorted_videos
        if video_sample:
            validation_log.append(f"Checking metadata in all {len(video_sample)} videos...")
    else:
        first_n = sorted_videos[:num_to_sample]
        last_n = sorted_videos[-num_to_sample:]
        video_sample = first_n + last_n
        validation_log.append(
            f"Checking metadata in {len(video_sample)} videos "
            f"(first {num_to_sample} and last {num_to_sample} sorted by filename) "
            f"out of {len(video_files)} total..."
        )

    # Add a random sample across all media on top of the first/last picks.
    # First/last-by-filename is ideal for a clean sequential single-camera
    # folder, but it misses two cases: (a) a handful of stripped-EXIF files
    # happening to sit at the filename extremes would make the whole folder
    # read as "no dates", and (b) a folder holding several camera subfolders
    # has no single meaningful filename order, so the extremes aren't
    # representative. A random draw covers both for ~100 extra EXIF reads
    # (well under a second). The date range this produces is a rough,
    # sample-based estimate, surfaced as date-only in the UI.
    random_sample_size = 100
    already_sampled = set(image_sample) | set(video_sample)
    pool = [f for f in (sorted_images + sorted_videos) if f not in already_sampled]
    if pool:
        extra = random.sample(pool, min(random_sample_size, len(pool)))
        image_sample = image_sample + [
            f for f in extra if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        video_sample = video_sample + [
            f for f in extra if f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        validation_log.append(
            f"Plus {len(extra)} randomly sampled file(s) across the folder."
        )

    # Extract dates from images
    validation_log.append("Images: Trying DateTimeOriginal → DateTimeDigitized → DateTime")
    image_dates = _extract_exif_dates(image_sample)

    # Extract dates from videos
    if video_sample:
        validation_log.append(
            "Videos: trying " + " → ".join(_VIDEO_DATE_FIELDS)
        )
        video_dates = _extract_video_dates(video_sample)
    else:
        video_dates = []

    # Combine all dates. Any timestamp found is used; the min/max give the
    # rough range the UI shows as a date-only estimate. There is no
    # minimum-span gate: the dates come from EXIF DateTimeOriginal (read
    # from the Exif sub-IFD), never from file mtime, so a narrow span is a
    # real narrow span, not the corrupt-mtime-cluster case the old 3-hour
    # rule guarded against.
    all_dates = image_dates + video_dates

    if all_dates:
        start_date, end_date = min(all_dates), max(all_dates)
        timespan_hours = (end_date - start_date).total_seconds() / 3600
        validation_log.append(
            f"✓ Found timestamps: {len(image_dates)} images + {len(video_dates)} videos "
            f"spanning {timespan_hours:.1f} hours ({start_date.strftime('%Y-%m-%d %H:%M')} to "
            f"{end_date.strftime('%Y-%m-%d %H:%M')})"
        )
        return start_date, end_date, validation_log

    validation_log.append("✗ No datetime metadata found in any images or videos")
    return None, None, validation_log


# DateTimeOriginal (36867) and DateTimeDigitized (36868) live in the Exif
# sub-IFD, reached via the pointer tag 0x8769 in the base IFD. They are NOT
# in the base IFD that Image.getexif() returns directly; reading them off
# the base IFD always yields None. DateTime (306) does live in the base IFD.
# Reading 36867/36868 off the base IFD was the bug that made camera-trap
# images with perfectly good capture times report "no datetime metadata".
# (The GPS reader above already does the equivalent get_ifd(0x8825) dance.)
_EXIF_IFD_POINTER = 0x8769
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_DATETIME = 306

# Trailing timezone designator: "Z", "+02:00", "-0500", etc. Observational
# datetimes are stored naive in camera-local wall-clock time (see
# DEVELOPERS.md "Datetime conventions"), so any offset is dropped, never
# applied.
_TZ_SUFFIX_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")

# strptime patterns tried in order. EXIF standard is colon-separated
# ("YYYY:MM:DD HH:MM:SS"); real cameras and re-encoders also emit dash or
# slash date separators, ISO "T", and date-only stamps.
_DATETIME_FORMATS = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y:%m:%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y:%m:%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y:%m:%d",
    "%Y-%m-%d",
)


def _parse_exif_datetime(raw: object) -> datetime | None:
    """Parse an EXIF date value into a naive datetime, tolerant of formats.

    Handles bytes or str, trailing NULs / whitespace, sub-second fractions,
    timezone suffixes (dropped — stored values are camera-local wall clock),
    and colon / dash / slash / ISO-"T" separators. The all-zero placeholder
    ("0000:00:00 ...") that re-encoders leave behind is rejected. Returns
    None when nothing parseable is found.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", "ignore")
    if not isinstance(raw, str):
        return None
    s = raw.replace("\x00", "").strip()
    if not s or s.startswith("0000:00:00") or s.startswith("0000-00-00"):
        return None
    # Drop a timezone suffix, then a sub-second fraction, leaving the bare
    # wall-clock components for the format table below.
    s = _TZ_SUFFIX_RE.sub("", s).strip()
    s = re.sub(r"\.\d+$", "", s)
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Last resort: normalise an EXIF "YYYY:MM:DD" date head to dashes and let
    # fromisoformat handle whatever time part remains.
    iso = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", s).replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(iso).replace(tzinfo=None)
    except ValueError:
        return None


def _read_exif_datetime(img: Image.Image) -> datetime | None:
    """Best capture datetime for an open image, or None.

    Reads DateTimeOriginal then DateTimeDigitized from the Exif sub-IFD
    (where they actually live), then DateTime from the base IFD. Also checks
    the base IFD for the first two in case a non-standard writer put them
    there. First parseable value wins.
    """
    exif = img.getexif()
    if not exif:
        return None
    sub = exif.get_ifd(_EXIF_IFD_POINTER) or {}
    for raw in (
        sub.get(_TAG_DATETIME_ORIGINAL),
        sub.get(_TAG_DATETIME_DIGITIZED),
        exif.get(_TAG_DATETIME_ORIGINAL),  # non-standard: in base IFD
        exif.get(_TAG_DATETIME_DIGITIZED),
        exif.get(_TAG_DATETIME),
    ):
        dt = _parse_exif_datetime(raw)
        if dt is not None:
            return dt
    return None


def _extract_exif_date_single(img_path: Path) -> datetime | None:
    """Extract the EXIF datetime from a single image. Returns None on failure."""
    try:
        with Image.open(img_path) as img:
            return _read_exif_datetime(img)
    except Exception:
        return None


def _extract_exif_dates(sample: list[Path]) -> list[datetime]:
    """
    Extract capture datetimes from a sample of images.

    Reads DateTimeOriginal / DateTimeDigitized from the Exif sub-IFD and
    DateTime from the base IFD (see ``_read_exif_datetime``), tolerant of
    the date formats real cameras emit (see ``_parse_exif_datetime``).
    """
    dates: list[datetime] = []

    for img_path in sample:
        try:
            with Image.open(img_path) as img:
                dt = _read_exif_datetime(img)
            if dt is not None:
                dates.append(dt)
        except Exception as e:
            logger.debug(f"Cannot read EXIF from {img_path.name}: {type(e).__name__}: {e}")
            continue

    return dates


def _extract_video_dates(sample: list[Path]) -> list[datetime]:
    """
    Extract dates from video metadata using exiftool.

    Delegates to the shared utility in app.utils.media_dates.

    Returns:
        List of datetime objects extracted from videos
    """
    date_map = _shared_extract_video_dates(sample)
    return list(date_map.values())
