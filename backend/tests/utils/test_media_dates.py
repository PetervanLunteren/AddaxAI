"""Tests for the two opt-in capture-date fallbacks.

The `addaxai-YYYYMMDD-HHMMSS` filename marker, and the file modification
time the user can opt into when a folder's metadata carries no dates.
"""

from __future__ import annotations

import os
from datetime import datetime

from app.utils.media_dates import (
    file_mtime_datetime,
    parse_addaxai_filename_datetime,
)


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
