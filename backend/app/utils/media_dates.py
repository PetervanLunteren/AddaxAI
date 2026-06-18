"""
Shared video date extraction using exiftool.

Extracts creation dates from video metadata. Used by both the folder scanner
(for deployment preview) and the JSON pipeline (for file timestamps).
"""

import re
from datetime import datetime
from pathlib import Path

import exiftool

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Strict, opt-in filename date fallback. A user whose files have no readable
# EXIF/metadata date can rename them to end with `addaxai-YYYYMMDD-HHMMSS`
# before the extension (e.g. `clip_addaxai-20250222-072314.mp4`); we parse the
# capture time from that. The `addaxai-` marker makes the match unambiguous (no
# false positives from serial numbers), so this is a silent last resort with no
# settings. The marker is case-insensitive; the separator is strictly `-`, and
# the block must end the filename stem.
_ADDAXAI_FILENAME_RE = re.compile(r"addaxai-(\d{8})-(\d{6})$", re.IGNORECASE)


def parse_addaxai_filename_datetime(filename: str) -> datetime | None:
    """Capture time from a `…addaxai-YYYYMMDD-HHMMSS.<ext>` filename, else None.

    Returns the naive local datetime, or None when the marker is absent or the
    digits are not a real calendar datetime.
    """
    match = _ADDAXAI_FILENAME_RE.search(Path(filename).stem)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None

# Metadata fields tried in order of preference
_DATE_FIELDS = [
    "QuickTime:CreateDate",
    "EXIF:DateTimeOriginal",
    "QuickTime:MediaCreateDate",
    "QuickTime:TrackCreateDate",
    "RIFF:DateTimeOriginal",
    "RIFF:DateCreated",
]

# Placeholder that container date fields hold when a tool (typically
# ffmpeg) re-encoded the video and never wrote a real timestamp. ExifTool
# returns the prefix "0000:00:00 00:00:00" verbatim. We treat zero-stamps
# as a parse failure so the loop keeps trying the remaining fields, and
# so the warning we log identifies the cause for the user.
_ZERO_STAMP_PREFIX = "0000:00:00 00:00:00"


def _parse_date_string(date_str: str) -> datetime | None:
    """Parse a date string from exiftool metadata, or None on failure.

    The all-zero placeholder is returned as None so the caller can fall
    through to the next field instead of accepting a garbage date.
    """
    if date_str.startswith(_ZERO_STAMP_PREFIX):
        return None
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
    Extract creation dates from multiple video files using a single
    exiftool process.

    Tries the fields in ``_DATE_FIELDS`` in order. The first field that
    parses to a valid datetime wins; malformed or zero-stamp values are
    skipped and the next field is tried. Files that yield nothing
    parseable are omitted from the returned dict and a warning is
    logged identifying the cause (no date fields at all, all fields
    zero-stamped, or unparseable values).
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
                        logger.warning(
                            f"No metadata returned by exiftool for "
                            f"{video_path.name}"
                        )
                        continue

                    metadata = metadata_list[0]

                    date_obj: datetime | None = None
                    fields_tried: list[tuple[str, str]] = []
                    for field in _DATE_FIELDS:
                        if field not in metadata:
                            continue
                        value = str(metadata[field])
                        fields_tried.append((field, value))
                        date_obj = _parse_date_string(value)
                        if date_obj is not None:
                            break

                    if date_obj is not None:
                        date_map[video_path] = date_obj
                        continue

                    if not fields_tried:
                        logger.warning(
                            f"No capture-date fields in {video_path.name}; "
                            f"tried {_DATE_FIELDS}"
                        )
                    elif all(
                        v.startswith(_ZERO_STAMP_PREFIX)
                        for _, v in fields_tried
                    ):
                        # FFMP in VendorID is ffmpeg's signature. When we see
                        # it alongside all-zero dates, the file was almost
                        # certainly re-encoded by ffmpeg and the original
                        # camera capture time was wiped from the container.
                        vendor = metadata.get("QuickTime:VendorID")
                        hint = (
                            " (re-encoded by ffmpeg, original capture date lost)"
                            if vendor == "FFMP"
                            else ""
                        )
                        logger.warning(
                            f"All date fields zero-stamped in "
                            f"{video_path.name}{hint}"
                        )
                    else:
                        logger.warning(
                            f"Unparseable date(s) in {video_path.name}: "
                            f"{fields_tried}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Cannot read metadata from {video_path.name}: "
                        f"{type(e).__name__}: {e}"
                    )
                    continue

    except Exception as e:
        logger.warning(f"ExifTool error: {type(e).__name__}: {e}")

    return date_map


def extract_video_date(path: Path) -> datetime | None:
    """Extract creation date from a single video file using exiftool."""
    date_map = extract_video_dates([path])
    return date_map.get(path)
