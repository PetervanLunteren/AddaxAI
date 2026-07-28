"""Tests for the opt-in `addaxai-YYYYMMDD-HHMMSS` filename date fallback."""

from __future__ import annotations

from datetime import datetime

from app.utils.media_dates import parse_addaxai_filename_datetime


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
