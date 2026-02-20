"""
Shared video date extraction using exiftool.

Extracts creation dates from video metadata. Used by both the folder scanner
(for deployment preview) and the JSON pipeline (for file timestamps).
"""

from datetime import datetime
from pathlib import Path

import exiftool

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Metadata fields tried in order of preference
_DATE_FIELDS = [
    "QuickTime:CreateDate",
    "EXIF:DateTimeOriginal",
    "QuickTime:MediaCreateDate",
    "QuickTime:TrackCreateDate",
]


def _parse_date_string(date_str: str) -> datetime | None:
    """Parse a date string from exiftool metadata."""
    try:
        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        try:
            date_str_clean = date_str.split("+")[0].split("-")[0].strip()
            return datetime.fromisoformat(date_str_clean)
        except ValueError:
            return None


def extract_video_dates(paths: list[Path]) -> dict[Path, datetime]:
    """
    Extract creation dates from multiple video files using a single exiftool process.

    Tries metadata fields in order of preference:
    1. QuickTime:CreateDate — most common for camera traps
    2. EXIF:DateTimeOriginal — some cameras use this
    3. QuickTime:MediaCreateDate — MP4 container metadata
    4. QuickTime:TrackCreateDate — video track metadata

    Files that yield no date are omitted from the returned dict.
    """
    if not paths:
        return {}

    date_map: dict[Path, datetime] = {}

    try:
        with exiftool.ExifToolHelper() as et:
            for video_path in paths:
                try:
                    metadata_list = et.get_metadata([str(video_path)])
                    if not metadata_list:
                        continue

                    metadata = metadata_list[0]

                    date_str = None
                    for field in _DATE_FIELDS:
                        if field in metadata:
                            date_str = metadata[field]
                            break

                    if date_str:
                        date_obj = _parse_date_string(date_str)
                        if date_obj:
                            date_map[video_path] = date_obj
                        else:
                            logger.debug(
                                f"Invalid date format in {video_path.name}: {date_str}"
                            )

                except Exception as e:
                    logger.debug(
                        f"Cannot read metadata from {video_path.name}: {type(e).__name__}: {e}"
                    )
                    continue

    except Exception as e:
        logger.warning(f"ExifTool error: {type(e).__name__}: {e}")

    return date_map


def extract_video_date(path: Path) -> datetime | None:
    """Extract creation date from a single video file using exiftool."""
    date_map = extract_video_dates([path])
    return date_map.get(path)
