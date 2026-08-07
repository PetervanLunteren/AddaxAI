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
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from PIL import Image
from PIL.ExifTags import GPSTAGS

from app.core.logging_config import get_logger
from app.core.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.utils.media_dates import _DATE_FIELDS as _VIDEO_DATE_FIELDS
from app.utils.media_dates import extract_video_dates as _shared_extract_video_dates
from app.utils.media_dates import file_mtime_datetime, read_exif_datetime

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


def prune_hidden_files(filenames: list[str]) -> list[str]:
    """Filter an ``os.walk`` file list down to non-hidden files.

    Drops dot-files: macOS AppleDouble ``._*`` sidecars (written next to
    real files on FAT-formatted SD cards), ``.DS_Store`` and friends. A
    sidecar carries a media extension, so without this it is counted,
    sampled in the Adjust-dates modal, and sent to the detector as if it
    were a real image or video. The file-side counterpart of
    ``prune_unscannable_dirs``, shared by the same walkers.
    """
    return [f for f in filenames if not f.startswith(".")]


def walk_media_files(folder: Path) -> tuple[list[Path], list[Path]]:
    """Every image and video under ``folder``, as ``(images, videos)``.

    Filenames only: nothing is opened and no metadata is read. Dot-folders
    and AddaxAI output folders are skipped via ``prune_unscannable_dirs``,
    dot-files via ``prune_hidden_files``.
    """
    images: list[Path] = []
    videos: list[Path] = []

    for root, dirs, files in os.walk(folder):
        dirs[:] = prune_unscannable_dirs(root, dirs)
        for filename in prune_hidden_files(files):
            file_path = Path(root) / filename
            ext = file_path.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(file_path)
            elif ext in VIDEO_EXTENSIONS:
                videos.append(file_path)

    return images, videos


def count_media_files(folder: Path) -> tuple[int, int]:
    """How many images and videos are under ``folder``, as ``(images, videos)``.

    The cheap counterpart to ``scan_folder``, which additionally reads EXIF
    and GPS from a sample of the files. The CSV import needs the counts for
    every folder in one request, so it cannot pay for the metadata reads.
    """
    images, videos = walk_media_files(folder)
    return len(images), len(videos)


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
    # Range the opt-in file-modification-time fallback would produce.
    # Non-null exactly when `missing_datetime` is True and the files could
    # be stat'ed, so the UI has no third state to handle.
    mtime_start_date: str | None  # ISO format datetime
    mtime_end_date: str | None  # ISO format datetime


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

    # Recursively find all media files. Dot-folders and AddaxAI output
    # folders are skipped inside the walk, so a previous run's separated /
    # visualised copies never get scanned back in as input media.
    image_files, video_files = walk_media_files(folder)

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

    # When nothing carries a capture date, work out what the opt-in
    # file-modification-time fallback would give, so the user can judge it
    # before ticking the box. That displayed range is the only safeguard
    # (there is no heuristic behind it), so it covers every file rather
    # than the sample the metadata reads use: a stat() is cheap where an
    # EXIF decode is not. Skipped entirely when real dates exist, so the
    # common scan never pays for the pass.
    mtime_start, mtime_end = (None, None)
    if start_date is None and all_media:
        mtime_start, mtime_end = _mtime_range(all_media)
        if mtime_start is not None and mtime_end is not None:
            validation_log.append(
                f"File modification times span {mtime_start:%Y-%m-%d %H:%M} "
                f"to {mtime_end:%Y-%m-%d %H:%M}"
            )

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
        mtime_start_date=mtime_start.isoformat() if mtime_start else None,
        mtime_end_date=mtime_end.isoformat() if mtime_end else None,
    )


def _mtime_range(files: list[Path]) -> tuple[datetime | None, datetime | None]:
    """Earliest and latest file modification time across ``files``.

    Both None when nothing could be stat'ed. Unreadable files are skipped
    rather than failing the scan; a preview that shows a slightly narrower
    range is more useful than no preview at all.
    """
    stamps = [ts for ts in (file_mtime_datetime(f) for f in files) if ts is not None]
    if not stamps:
        return None, None
    return min(stamps), max(stamps)


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
    # from the Exif sub-IFD) and never from file mtime, which `scan_folder`
    # reports separately and only when this function finds nothing. So a
    # narrow span here is a real narrow span, not the corrupt-mtime-cluster
    # case the old 3-hour rule guarded against.
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


def _extract_exif_dates(sample: list[Path]) -> list[datetime]:
    """
    Extract capture datetimes from a sample of images.

    Reads DateTimeOriginal / DateTimeDigitized from the Exif sub-IFD and
    DateTime from the base IFD, tolerant of the date formats real cameras
    emit. The reader lives in ``app.utils.media_dates`` and is shared with
    the JSON pipeline, so the dates this preview reports are the dates
    ingest stores.
    """
    dates: list[datetime] = []

    for img_path in sample:
        try:
            with Image.open(img_path) as img:
                dt = read_exif_datetime(img)
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
