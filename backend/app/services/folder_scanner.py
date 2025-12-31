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

import random
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class GPSCoordinates(TypedDict):
    """GPS coordinates extracted from EXIF."""

    latitude: float
    longitude: float


class FolderPreview(TypedDict):
    """Preview of a deployment folder."""

    image_count: int
    video_count: int
    total_count: int
    gps_location: GPSCoordinates | None
    sample_files: list[str]  # Relative paths
    start_date: str | None  # ISO format datetime
    end_date: str | None  # ISO format datetime
    missing_datetime: bool  # True if no EXIF dates found
    datetime_validation_log: list[str]  # Log of what was tried and why rejected


# Supported file extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".gif", ".png", ".tif", ".tiff", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv")


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

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            image_files.append(file_path)
        elif ext in VIDEO_EXTENSIONS:
            video_files.append(file_path)

    # Get relative paths for sample
    sample_files = [
        str(f.relative_to(folder))
        for f in (image_files[:5] + video_files[:2])[:10]
    ]

    # Try to extract GPS from random sample of images
    gps_location = _extract_gps_from_sample(folder, image_files, gps_sample_size)

    # Extract date range from images with validation
    start_date, end_date, validation_log = _extract_date_range(image_files)

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
            logger.warning(
                f"Failed to extract GPS from {img_path.name}: {type(e).__name__}: {e}"
            )
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
            lat = _convert_to_degrees(
                gps_data.get("GPSLatitude"), gps_data.get("GPSLatitudeRef")
            )
            lon = _convert_to_degrees(
                gps_data.get("GPSLongitude"), gps_data.get("GPSLongitudeRef")
            )

            if lat is not None and lon is not None:
                return GPSCoordinates(latitude=lat, longitude=lon)

            return None

    except Exception as e:
        # Image corrupt, not readable, or other error - log it
        logger.debug(
            f"Cannot read image {img_path.name}: {type(e).__name__}: {e}"
        )
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
) -> tuple[datetime | None, datetime | None, list[str]]:
    """
    Extract date range from image EXIF datetime metadata with validation.

    Tries EXIF date tags in order of preference:
    1. DateTimeOriginal (36867) - camera capture time
    2. DateTimeDigitized (36868) - when digitized
    3. DateTime (306) - file modification time in camera

    Validates that date range is at least 6 hours (filters out invalid/corrupt timestamps).

    Checks first 50 and last 50 images (sorted by filename) for date range.
    Camera traps use sequential filenames (IMG_0001.jpg, IMG_0002.jpg, etc.)
    so first and last files give accurate min/max dates.

    Returns:
        Tuple of (start_date, end_date, validation_log)
        validation_log contains human-readable messages about what was tried
    """
    if not image_files:
        return None, None, ["No image files found"]

    validation_log: list[str] = []

    # Minimum valid timespan: 6 hours
    # Catches file modification/creation times which cluster together
    MIN_TIMESPAN_HOURS = 6

    # Sort by filename and sample first/last
    # Camera traps use sequential filenames, so this gives us chronological order
    sorted_files = sorted(image_files, key=lambda p: p.name)

    # Take first 50 and last 50 (or whatever is available)
    num_to_sample = min(50, len(sorted_files))
    if len(sorted_files) <= 100:
        # If we have 100 or fewer images, just check them all
        sample = sorted_files
        validation_log.append(f"Checking EXIF metadata in all {len(sample)} images...")
    else:
        # Take first 50 and last 50
        first_n = sorted_files[:num_to_sample]
        last_n = sorted_files[-num_to_sample:]
        sample = first_n + last_n
        validation_log.append(
            f"Checking EXIF metadata in {len(sample)} images "
            f"(first {num_to_sample} and last {num_to_sample} sorted by filename) "
            f"out of {len(image_files)} total..."
        )

    validation_log.append("Trying: DateTimeOriginal → DateTimeDigitized → DateTime")
    exif_dates = _extract_exif_dates(sample)

    if exif_dates:
        start_date, end_date = min(exif_dates), max(exif_dates)
        timespan_hours = (end_date - start_date).total_seconds() / 3600

        if timespan_hours >= MIN_TIMESPAN_HOURS:
            validation_log.append(
                f"✓ Found valid EXIF dates: {len(exif_dates)} images spanning "
                f"{timespan_hours:.1f} hours ({start_date.strftime('%Y-%m-%d %H:%M')} to "
                f"{end_date.strftime('%Y-%m-%d %H:%M')})"
            )
            return start_date, end_date, validation_log
        else:
            validation_log.append(
                f"✗ EXIF dates rejected: Only {timespan_hours:.1f} hours span "
                f"(minimum {MIN_TIMESPAN_HOURS} hours required). "
                f"This likely indicates corrupt timestamps or images all taken at the same time."
            )
    else:
        validation_log.append("✗ No EXIF datetime metadata found in any images")

    validation_log.append(
        "✗ DateTime extraction failed - no valid EXIF timestamps found"
    )
    return None, None, validation_log


def _extract_exif_dates(sample: list[Path]) -> list[datetime]:
    """
    Extract dates from EXIF metadata.

    Tries EXIF tags in order of preference:
    1. DateTimeOriginal (36867) - camera capture time, most accurate
    2. DateTimeDigitized (36868) - when digitized
    3. DateTime (306) - file modification time in camera
    """
    dates: list[datetime] = []

    for img_path in sample:
        try:
            with Image.open(img_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    continue

                # Try date tags in order of preference
                date_str = exif_data.get(36867)  # DateTimeOriginal
                if not date_str:
                    date_str = exif_data.get(36868)  # DateTimeDigitized
                if not date_str:
                    date_str = exif_data.get(306)  # DateTime

                if date_str:
                    # Parse EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                    try:
                        date_obj = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                        dates.append(date_obj)
                    except ValueError:
                        logger.debug(f"Invalid date format in {img_path.name}: {date_str}")
                        continue

        except Exception as e:
            logger.debug(
                f"Cannot read EXIF from {img_path.name}: {type(e).__name__}: {e}"
            )
            continue

    return dates
