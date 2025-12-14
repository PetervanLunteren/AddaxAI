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
from pathlib import Path
from typing import TypedDict

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


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

    return FolderPreview(
        image_count=len(image_files),
        video_count=len(video_files),
        total_count=len(image_files) + len(video_files),
        gps_location=gps_location,
        sample_files=sample_files,
    )


def _extract_gps_from_sample(
    folder: Path, image_files: list[Path], sample_size: int
) -> GPSCoordinates | None:
    """
    Extract GPS coordinates from a random sample of images.

    Checks up to sample_size random images. If GPS found in multiple,
    averages the coordinates.

    Returns:
        Average GPS coordinates or None if not found
    """
    if not image_files:
        return None

    # Random sample (don't exceed available files)
    sample = random.sample(image_files, min(sample_size, len(image_files)))

    gps_coords: list[GPSCoordinates] = []

    for img_path in sample:
        try:
            coords = _extract_gps_from_image(img_path)
            if coords:
                gps_coords.append(coords)
        except Exception:
            # Skip files with corrupt EXIF or other issues
            continue

        # If we found GPS in a few files, we can stop early
        if len(gps_coords) >= 3:
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

            # Find GPS IFD tag
            gps_ifd = None
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    gps_ifd = value
                    break

            if not gps_ifd:
                return None

            # Parse GPS data
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

    except Exception:
        # Image corrupt, not readable, or other error
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
