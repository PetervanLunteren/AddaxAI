"""Tests for the ingest-side capture-timestamp ladder.

The folder scan reads the full EXIF tag ladder (DateTimeOriginal →
DateTimeDigitized → DateTime) off the image itself, but the detection
JSON's exif_metadata block only carries what the detector extracted
(DateTimeOriginal). Ingest therefore re-reads the file with the same
shared reader when the JSON yields nothing, so the dates the preview
promises are the dates the database gets. These tests pin that ladder
and its ordering against the filename-marker and mtime fallbacks.
"""

import os
from datetime import datetime

from PIL import Image

from app.ml.json_pipeline import _resolve_capture_timestamp


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


def _resolve(path, exif_metadata=None, use_mtime=False):
    return _resolve_capture_timestamp(
        path,
        is_video=False,
        exif_metadata=exif_metadata,
        video_dates={},
        use_file_mtime_fallback=use_mtime,
    )


def test_json_datetime_original_wins_without_touching_the_file(tmp_path):
    # The path deliberately does not exist: when the JSON has the date,
    # the file must not be opened.
    ts, source = _resolve(
        tmp_path / "missing.jpg",
        exif_metadata={"DateTimeOriginal": "2024:06:15 09:00:00"},
    )
    assert ts == datetime(2024, 6, 15, 9, 0, 0)
    assert source == "metadata"


def test_json_fallback_tags_and_tolerant_formats_parse(tmp_path):
    ts, source = _resolve(
        tmp_path / "missing.jpg",
        exif_metadata={"DateTime": "2024-06-15 09:00:00"},
    )
    assert ts == datetime(2024, 6, 15, 9, 0, 0)
    assert source == "metadata"


def test_file_reread_recovers_base_ifd_datetime(tmp_path):
    """A camera that writes only the plain DateTime tag: the detection
    JSON carries no date, but the image itself does. Ingest must find it,
    exactly like the folder scan did at preview time."""
    p = tmp_path / "IMG_0001.jpg"
    _write_jpeg(p, datetime_tag="2024:03:15 08:00:00")
    ts, source = _resolve(p, exif_metadata=None)
    assert ts == datetime(2024, 3, 15, 8, 0, 0)
    assert source == "exif_reread"


def test_file_exif_outranks_addaxai_filename_marker(tmp_path):
    """Real EXIF beats the filename convention: a separated copy that
    kept its EXIF must use it, not the name it was given."""
    p = tmp_path / "clip_addaxai-20250222-072314.jpg"
    _write_jpeg(p, datetime_original="2024:03:15 08:00:00")
    ts, source = _resolve(p, exif_metadata=None)
    assert ts == datetime(2024, 3, 15, 8, 0, 0)
    assert source == "exif_reread"


def test_filename_marker_used_when_file_has_no_exif(tmp_path):
    p = tmp_path / "clip_addaxai-20250222-072314.jpg"
    _write_jpeg(p)
    ts, source = _resolve(p, exif_metadata=None)
    assert ts == datetime(2025, 2, 22, 7, 23, 14)
    assert source == "filename"


def test_mtime_stays_last_and_opt_in(tmp_path):
    p = tmp_path / "IMG_0001.jpg"
    _write_jpeg(p)
    stamp = datetime(2024, 5, 1, 10, 30, 0).timestamp()
    os.utime(p, (stamp, stamp))

    ts, source = _resolve(p, exif_metadata=None, use_mtime=False)
    assert ts is None
    assert source == "none"

    ts, source = _resolve(p, exif_metadata=None, use_mtime=True)
    assert ts == datetime(2024, 5, 1, 10, 30, 0)
    assert source == "mtime"
