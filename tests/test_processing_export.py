"""Tests for addaxai.processing.export — data export utilities."""

import datetime
import hashlib
import json
import os
import pytest

from addaxai.processing.export import (
    CSV_DTYPES,
    clean_line,
    format_datetime,
    generate_unique_id,
)


# --- clean_line ---

def test_clean_line_removes_null_bytes():
    assert clean_line("hello\0world") == "helloworld"


def test_clean_line_no_nulls():
    assert clean_line("normal text") == "normal text"


# --- generate_unique_id ---

def test_generate_unique_id_returns_md5():
    row = ["deer", "0.95", "photo.jpg"]
    result = generate_unique_id(row)
    expected = hashlib.md5("deer0.95photo.jpg".encode("utf-8")).hexdigest()
    assert result == expected


def test_generate_unique_id_different_rows_differ():
    row1 = ["deer", "photo1.jpg"]
    row2 = ["bear", "photo2.jpg"]
    assert generate_unique_id(row1) != generate_unique_id(row2)


# --- format_datetime ---

def test_format_datetime_valid():
    result = format_datetime("15/03/22 12:30:45")
    assert result == "2022-03-15T12:30:45"


def test_format_datetime_invalid():
    assert format_datetime("not-a-date") == "NA"


def test_format_datetime_na():
    assert format_datetime("NA") == "NA"


# --- CSV_DTYPES ---

def test_csv_dtypes_is_dict():
    assert isinstance(CSV_DTYPES, dict)
    assert "relative_path" in CSV_DTYPES
    assert "confidence" in CSV_DTYPES
    assert CSV_DTYPES["confidence"] == "float64"


# --- csv_to_coco (requires pandas) ---

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas not installed")
def test_csv_to_coco_basic(tmp_path):
    from addaxai.processing.export import csv_to_coco

    detections_df = pd.DataFrame({
        "relative_path": ["img1.jpg", "img1.jpg"],
        "label": ["deer", "bear"],
        "bbox_left": [10, 50],
        "bbox_top": [20, 60],
        "bbox_right": [100, 150],
        "bbox_bottom": [200, 250],
    })
    files_df = pd.DataFrame({
        "relative_path": ["img1.jpg"],
        "file_width": [640],
        "file_height": [480],
        "DateTimeOriginal": ["15/03/22 12:30:45"],
    })
    output = tmp_path / "coco.json"
    csv_to_coco(detections_df, files_df, str(output), version="1.0")

    data = json.loads(output.read_text())
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 2
    assert len(data["categories"]) == 2
    assert "AddaxAI" in data["info"]["description"]


@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas not installed")
def test_csv_to_coco_na_date(tmp_path):
    from addaxai.processing.export import csv_to_coco

    detections_df = pd.DataFrame({
        "relative_path": ["img1.jpg"],
        "label": ["deer"],
        "bbox_left": [10],
        "bbox_top": [20],
        "bbox_right": [100],
        "bbox_bottom": [200],
    })
    files_df = pd.DataFrame({
        "relative_path": ["img1.jpg"],
        "file_width": [640],
        "file_height": [480],
        "DateTimeOriginal": [float("nan")],  # NA value
    })
    output = tmp_path / "coco.json"
    csv_to_coco(detections_df, files_df, str(output))

    data = json.loads(output.read_text())
    assert data["images"][0]["date_captured"] == "NA"
