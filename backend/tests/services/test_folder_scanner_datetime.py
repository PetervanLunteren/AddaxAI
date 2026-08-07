"""Tests for EXIF capture-datetime extraction in the folder scanner.

DateTimeOriginal / DateTimeDigitized live in the Exif sub-IFD (pointer tag
0x8769), not the base IFD that Image.getexif() returns. Reading them off the
base IFD silently yields None, which made camera-trap images with valid
capture times report "no datetime metadata" in the folder preview. These
tests pin the sub-IFD read and the tolerant date parser.
"""

import os
from datetime import datetime

import pytest
from PIL import Image

from app.services.folder_scanner import scan_folder
from app.utils.media_dates import (
    extract_image_date,
    parse_exif_datetime,
    read_exif_datetime,
)


def _write_jpeg(path, *, datetime_original=None, datetime_tag=None) -> None:
    """Write a tiny JPEG. datetime_original goes in the Exif sub-IFD
    (tag 0x8769 → 36867), datetime_tag in the base IFD (306)."""
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    exif = Image.Exif()
    if datetime_tag is not None:
        exif[306] = datetime_tag
    if datetime_original is not None:
        exif[0x8769] = {36867: datetime_original}
    img.save(path, format="JPEG", exif=exif)


# ── parse_exif_datetime ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2016:06:13 14:21:45", datetime(2016, 6, 13, 14, 21, 45)),
        ("2016-06-13 14:21:45", datetime(2016, 6, 13, 14, 21, 45)),
        ("2016/06/13 14:21:45", datetime(2016, 6, 13, 14, 21, 45)),
        ("2016:06:13T14:21:45", datetime(2016, 6, 13, 14, 21, 45)),
        # sub-second fraction dropped
        ("2016:06:13 14:21:45.123", datetime(2016, 6, 13, 14, 21, 45)),
        # timezone suffix dropped (stored naive, camera-local wall clock)
        ("2016:06:13 14:21:45+02:00", datetime(2016, 6, 13, 14, 21, 45)),
        ("2016:06:13 14:21:45Z", datetime(2016, 6, 13, 14, 21, 45)),
        ("2016:06:13 14:21:45-0500", datetime(2016, 6, 13, 14, 21, 45)),
        # minute precision and date-only
        ("2016:06:13 14:21", datetime(2016, 6, 13, 14, 21, 0)),
        ("2016:06:13", datetime(2016, 6, 13, 0, 0, 0)),
        # bytes with trailing NUL
        (b"2016:06:13 14:21:45\x00", datetime(2016, 6, 13, 14, 21, 45)),
    ],
)
def test_parse_exif_datetime_formats(raw, expected):
    assert parse_exif_datetime(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "0000:00:00 00:00:00",  # re-encoder zero stamp
        "0000-00-00 00:00:00",
        "garbage",
        12345,  # wrong type
    ],
)
def test_parse_exif_datetime_rejects(raw):
    assert parse_exif_datetime(raw) is None


# ── read_exif_datetime / extract_image_date ───────────────────────


def test_reads_datetime_original_from_sub_ifd(tmp_path):
    """The core bug: DateTimeOriginal in the sub-IFD must be found."""
    p = tmp_path / "img.jpg"
    _write_jpeg(p, datetime_original="2016:06:13 14:21:45")
    with Image.open(p) as img:
        assert read_exif_datetime(img) == datetime(2016, 6, 13, 14, 21, 45)
    assert extract_image_date(p) == datetime(2016, 6, 13, 14, 21, 45)


def test_falls_back_to_base_ifd_datetime(tmp_path):
    """No sub-IFD original, but DateTime (306) in the base IFD is used."""
    p = tmp_path / "img.jpg"
    _write_jpeg(p, datetime_tag="2016:06:13 09:00:00")
    assert extract_image_date(p) == datetime(2016, 6, 13, 9, 0, 0)


def test_no_exif_returns_none(tmp_path):
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(p, format="JPEG")
    assert extract_image_date(p) is None


# ── scan_folder integration ───────────────────────────────────────────────


def test_scan_folder_reports_dates_from_sub_ifd(tmp_path):
    """End-to-end: a folder of sub-IFD-dated images is NOT flagged as
    missing datetime, and the range spans the two timestamps."""
    _write_jpeg(tmp_path / "a.jpg", datetime_original="2016:06:13 08:00:00")
    _write_jpeg(tmp_path / "b.jpg", datetime_original="2016:06:13 18:30:00")

    preview = scan_folder(str(tmp_path))

    assert preview["image_count"] == 2
    assert preview["missing_datetime"] is False
    assert preview["start_date"] == datetime(2016, 6, 13, 8, 0, 0).isoformat()
    assert preview["end_date"] == datetime(2016, 6, 13, 18, 30, 0).isoformat()


def test_scan_folder_random_sample_rescues_when_extremes_lack_exif(tmp_path):
    """The first-5 + last-5 by filename can all lack EXIF while files in
    the middle carry dates. The random sample (size 100 >> the handful of
    remaining files here, so it deterministically covers the whole pool)
    must still surface those dates instead of reporting "no datetime"."""
    # 12 images: the 10 filename-extremes (01-05, 08-12) have no EXIF; the
    # two middle ones (06, 07) carry dates only reachable via the random
    # sample.
    for i in range(1, 13):
        name = f"{i:02d}.jpg"
        if i == 6:
            _write_jpeg(tmp_path / name, datetime_original="2016:06:13 08:00:00")
        elif i == 7:
            _write_jpeg(tmp_path / name, datetime_original="2016:06:13 18:00:00")
        else:
            Image.new("RGB", (4, 4), (1, 2, 3)).save(tmp_path / name, format="JPEG")

    preview = scan_folder(str(tmp_path))

    assert preview["image_count"] == 12
    assert preview["missing_datetime"] is False
    assert preview["start_date"] == datetime(2016, 6, 13, 8, 0, 0).isoformat()
    assert preview["end_date"] == datetime(2016, 6, 13, 18, 0, 0).isoformat()


def test_scan_folder_accepts_short_span(tmp_path):
    """No minimum-span gate: dates only minutes apart are still reported.
    `start_date` / `end_date` come from EXIF DateTimeOriginal and never
    from file mtime (which the scan reports separately, and only when no
    capture dates exist at all), so a narrow span is a real one, not the
    corrupt-mtime case the old 3-hour rule rejected."""
    _write_jpeg(tmp_path / "a.jpg", datetime_original="2016:06:13 08:00:00")
    _write_jpeg(tmp_path / "b.jpg", datetime_original="2016:06:13 08:30:00")

    preview = scan_folder(str(tmp_path))

    assert preview["missing_datetime"] is False
    assert preview["start_date"] == datetime(2016, 6, 13, 8, 0, 0).isoformat()
    assert preview["end_date"] == datetime(2016, 6, 13, 8, 30, 0).isoformat()


# ── file-modification-time range ──────────────────────────────────────────


def test_scan_folder_skips_mtime_range_when_dates_exist(tmp_path):
    """The stat pass is not just correct, it does not run at all when the
    files carry real capture dates. That laziness is what keeps the common
    scan free on slow SD readers and network drives."""
    _write_jpeg(tmp_path / "a.jpg", datetime_original="2016:06:13 08:00:00")
    _write_jpeg(tmp_path / "b.jpg", datetime_original="2016:06:13 18:30:00")

    preview = scan_folder(str(tmp_path))

    assert preview["missing_datetime"] is False
    assert preview["mtime_start_date"] is None
    assert preview["mtime_end_date"] is None


def test_scan_folder_reports_mtime_range_when_no_dates(tmp_path):
    """With no capture dates anywhere, the scan offers the exact range the
    opt-in fallback would produce. It covers EVERY file, not the sampled
    subset the EXIF reads use: the displayed range is the only safeguard
    the user gets, so it must not be an estimate. The outlier below sits
    outside the first-5 / last-5 filename window on purpose."""
    for i in range(1, 13):
        target = tmp_path / f"{i:02d}.jpg"
        Image.new("RGB", (4, 4), (1, 2, 3)).save(target, format="JPEG")
        # File 07 is neither in the first 5 nor the last 5 by filename.
        stamp = (
            datetime(2024, 4, 1, 9, 0, 0)
            if i == 7
            else datetime(2024, 4, 8, 12, 0, 0)
        ).timestamp()
        os.utime(target, (stamp, stamp))

    preview = scan_folder(str(tmp_path))

    assert preview["missing_datetime"] is True
    assert preview["mtime_start_date"] == datetime(2024, 4, 1, 9, 0, 0).isoformat()
    assert preview["mtime_end_date"] == datetime(2024, 4, 8, 12, 0, 0).isoformat()
    assert any(
        "File modification times span" in line
        for line in preview["datetime_validation_log"]
    )


def test_scan_folder_empty_folder_has_no_mtime_range(tmp_path):
    """Nothing to stat, and no crash."""
    preview = scan_folder(str(tmp_path))

    assert preview["total_count"] == 0
    assert preview["mtime_start_date"] is None
    assert preview["mtime_end_date"] is None
