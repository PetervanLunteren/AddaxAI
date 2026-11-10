"""Tests for the video date ladder and the two opt-in capture-date fallbacks.

The exiftool video reader, the `addaxai-YYYYMMDD-HHMMSS` filename marker,
and the file modification time the user can opt into when a folder's
metadata carries no dates.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.utils.media_dates import (
    extract_video_dates,
    file_mtime_datetime,
    parse_addaxai_filename_datetime,
)

# ── video dates via exiftool ──────────────────────────────────────────────
#
# The video ladder has to parse what exiftool prints, and with the `-G`
# grouping pyexiftool uses, Apple's Keys:CreationDate and Microsoft's Xtra
# EncodingTime arrive under the QuickTime prefix with a timezone suffix.
# Until 2026-10 the video path had its own parser that returned None on
# any suffix, so such a value could never be read. It now shares
# `parse_exif_datetime` with the image path.


def _video_dates(metadata: dict[str, str]) -> dict[Path, datetime]:
    """Run `extract_video_dates` on one fake clip with the given exiftool
    output, without an exiftool binary."""
    clip = Path("/clips/VID_0001.mp4")
    helper = MagicMock()
    helper.__enter__.return_value = helper
    helper.get_metadata.return_value = [metadata]
    with (
        patch("app.utils.media_dates.resolve_exiftool", return_value="exiftool"),
        patch("app.utils.media_dates.exiftool.ExifToolHelper", return_value=helper),
    ):
        return extract_video_dates([clip])


def test_plain_create_date_parses():
    dates = _video_dates({"QuickTime:CreateDate": "2026:09:30 14:03:22"})
    assert list(dates.values()) == [datetime(2026, 9, 30, 14, 3, 22)]


def test_utc_suffix_is_dropped_not_applied():
    dates = _video_dates({"QuickTime:CreateDate": "2026:09:30 14:03:22Z"})
    assert list(dates.values()) == [datetime(2026, 9, 30, 14, 3, 22)]


def test_offset_suffix_keeps_the_wall_clock():
    """A Keys-style value is the camera's local time plus its offset. The
    stored value is the wall clock, so the offset is dropped, never added."""
    dates = _video_dates({"QuickTime:CreateDate": "2026:09:30 10:03:22-04:00"})
    assert list(dates.values()) == [datetime(2026, 9, 30, 10, 3, 22)]


def test_zero_stamp_falls_through_to_the_next_field():
    dates = _video_dates(
        {
            "QuickTime:CreateDate": "0000:00:00 00:00:00",
            "QuickTime:MediaCreateDate": "2026:09:30 14:03:22",
        }
    )
    assert list(dates.values()) == [datetime(2026, 9, 30, 14, 3, 22)]


def test_only_zero_stamps_yields_no_date():
    assert _video_dates({"QuickTime:CreateDate": "0000:00:00 00:00:00"}) == {}


# ── filename marker ───────────────────────────────────────────────────────


def test_parses_marked_filename():
    assert parse_addaxai_filename_datetime(
        "S1_clip_addaxai-20250222-072314.mp4"
    ) == datetime(2025, 2, 22, 7, 23, 14)


def test_marker_is_case_insensitive():
    assert parse_addaxai_filename_datetime(
        "ADDAXAI-20250222-072314.JPG"
    ) == datetime(2025, 2, 22, 7, 23, 14)


def test_no_marker_returns_none():
    # Looks like a date, but no addaxai marker -> ignored (no false positives).
    assert parse_addaxai_filename_datetime("S1_20250222_072314.mp4") is None


def test_must_end_the_stem():
    assert (
        parse_addaxai_filename_datetime("addaxai-20250222-072314_edited.mp4") is None
    )


def test_strict_hyphen_separator():
    assert parse_addaxai_filename_datetime("addaxai_20250222_072314.mp4") is None


def test_invalid_calendar_date_returns_none():
    # Month 13 is not a real date.
    assert parse_addaxai_filename_datetime("addaxai-20251301-072314.mp4") is None


# ── file modification time ────────────────────────────────────────────────


def test_file_mtime_returns_naive_local_time(tmp_path):
    """The value the OS file browser shows, as a naive local datetime."""
    target = tmp_path / "IMG_0001.AVI"
    target.write_bytes(b"x")
    expected = datetime(2024, 4, 7, 15, 55, 26)
    os.utime(target, (expected.timestamp(), expected.timestamp()))

    result = file_mtime_datetime(target)

    assert result == expected
    assert result.tzinfo is None


def test_file_mtime_truncates_microseconds(tmp_path):
    """Every other capture-date source is second-resolution; match it."""
    target = tmp_path / "IMG_0002.AVI"
    target.write_bytes(b"x")
    stamp = datetime(2024, 4, 7, 15, 55, 26).timestamp() + 0.75
    os.utime(target, (stamp, stamp))

    assert file_mtime_datetime(target).microsecond == 0


def test_file_mtime_missing_file_returns_none(tmp_path):
    """A file can vanish between the detection run and the database load."""
    assert file_mtime_datetime(tmp_path / "gone.jpg") is None
