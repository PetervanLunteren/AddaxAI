"""
Shared capture-date extraction.

Holds the image EXIF reader, the exiftool video reader, and the two
opt-in last-resort fallbacks (filename marker, filesystem mtime). Used by
the folder scanner (deployment preview), the per-file probe endpoint, and
the JSON pipeline (file timestamps), so all three agree on where a
capture date can come from and what the preview promises is what ingest
stores.
"""

import re
from datetime import datetime
from pathlib import Path

import exiftool
from PIL import Image

from app.core.logging_config import get_logger
from app.utils.exiftool_bin import resolve_exiftool

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


def file_mtime_datetime(path: Path) -> datetime | None:
    """Filesystem mtime as a naive local datetime, or None if unreadable.

    This is the computer's own wall clock, the same value Finder or File
    Explorer shows, and the same value the folder scan shows the user
    before they opt in. Truncated to whole seconds because every other
    capture-date source is second-resolution.

    Opt-in only, never a silent default: see `use_file_mtime_fallback` on
    DeploymentQueue and the datetime conventions in DEVELOPERS.md. The
    value is right only while the files have not been copied; copying can
    reset it to the copy date or shift it by whole hours.
    """
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0)
    except OSError:
        # Matches _safe_file_size in json_pipeline: a file can disappear
        # between the detection run and the database load.
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

    # Resolved outside the tolerant try-block below: a missing exiftool
    # binary is an installation problem and must fail loudly, unlike
    # per-file metadata trouble which is tolerated.
    executable = resolve_exiftool()

    try:
        with exiftool.ExifToolHelper(executable=executable) as et:
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


# ── Image EXIF dates ──────────────────────────────────────────────────────
#
# DateTimeOriginal (36867) and DateTimeDigitized (36868) live in the Exif
# sub-IFD, reached via the pointer tag 0x8769 in the base IFD. They are NOT
# in the base IFD that Image.getexif() returns directly; reading them off
# the base IFD always yields None. DateTime (306) does live in the base IFD.
# Reading 36867/36868 off the base IFD was the bug that made camera-trap
# images with perfectly good capture times report "no datetime metadata".
_EXIF_IFD_POINTER = 0x8769
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_DATETIME = 306

# The same ladder by key name, for callers holding an already-extracted
# EXIF dict (MegaDetector's `exif_metadata` block) instead of an open
# image. Priority order matters: DateTimeOriginal is the capture moment,
# DateTimeDigitized the digitisation moment, DateTime the in-camera file
# write. The weaker tags are only consulted when the stronger are absent,
# so an edited image whose DateTime holds the edit moment still resolves
# to its DateTimeOriginal.
EXIF_DATE_KEYS = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]

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


def parse_exif_datetime(raw: object) -> datetime | None:
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


def read_exif_datetime(img: Image.Image) -> datetime | None:
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
        dt = parse_exif_datetime(raw)
        if dt is not None:
            return dt
    return None


def extract_image_date(img_path: Path) -> datetime | None:
    """Extract the EXIF datetime from a single image. Returns None on failure."""
    try:
        with Image.open(img_path) as img:
            return read_exif_datetime(img)
    except Exception:
        return None


def date_from_exif_dict(exif_metadata: dict | None) -> datetime | None:
    """Best capture datetime from an already-extracted EXIF dict, or None.

    Same tag ladder as `read_exif_datetime`, keyed by name, for the
    `exif_metadata` block a detection JSON carries per image.
    """
    if not exif_metadata:
        return None
    for key in EXIF_DATE_KEYS:
        dt = parse_exif_datetime(exif_metadata.get(key))
        if dt is not None:
            return dt
    return None
