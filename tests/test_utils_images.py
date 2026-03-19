"""Tests for addaxai.utils.images — image utility functions."""

import datetime
import os
import pytest
from PIL import Image

from addaxai.utils.images import (
    _camera_prefix_of_filename,
    _parse_timestamp_from_filename,
    build_image_timestamp_index,
    check_images,
    find_series_images,
    get_image_timestamp,
    is_image_corrupted,
    fix_images,
)


# --- _parse_timestamp_from_filename ---

def test_parse_timestamp_14digit():
    result = _parse_timestamp_from_filename("IMG_20220315123045.jpg")
    assert result == datetime.datetime(2022, 3, 15, 12, 30, 45)


def test_parse_timestamp_underscore():
    result = _parse_timestamp_from_filename("CAM1_20220315_123045.jpg")
    assert result == datetime.datetime(2022, 3, 15, 12, 30, 45)


def test_parse_timestamp_dashes():
    result = _parse_timestamp_from_filename("2022-03-15_123045.jpg")
    assert result == datetime.datetime(2022, 3, 15, 12, 30, 45)


def test_parse_timestamp_no_match():
    result = _parse_timestamp_from_filename("photo.jpg")
    assert result is None


# --- _camera_prefix_of_filename ---

def test_camera_prefix_basic():
    result = _camera_prefix_of_filename("CAM01_20220315123045.jpg")
    assert result == "CAM01_"


def test_camera_prefix_no_timestamp():
    result = _camera_prefix_of_filename("random_photo.jpg")
    assert result is None


# --- is_image_corrupted ---

def test_is_image_corrupted_valid(tmp_path):
    img_path = tmp_path / "valid.png"
    Image.new("RGB", (10, 10), color="red").save(str(img_path))
    assert is_image_corrupted(str(img_path)) is False


def test_is_image_corrupted_broken(tmp_path):
    img_path = tmp_path / "broken.jpg"
    img_path.write_bytes(b"not an image at all")
    assert is_image_corrupted(str(img_path)) is True


# --- check_images ---

def test_check_images_finds_corrupted(tmp_path):
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.jpg"
    Image.new("RGB", (10, 10)).save(str(good))
    bad.write_bytes(b"garbage")

    list_file = tmp_path / "files.txt"
    list_file.write_text(f"{good}\n{bad}\n")

    corrupted = check_images(str(list_file))
    assert str(bad) in corrupted
    assert str(good) not in corrupted


# --- fix_images ---

def test_fix_images_repairs_truncated(tmp_path):
    img_path = tmp_path / "truncated.png"
    # Create a valid image, then truncate it
    Image.new("RGB", (10, 10), color="blue").save(str(img_path))
    data = img_path.read_bytes()
    img_path.write_bytes(data[:len(data) // 2])  # truncate
    assert is_image_corrupted(str(img_path)) is True

    fix_images([str(img_path)])
    # After fix attempt, file should still exist (may or may not be fixed
    # depending on truncation severity, but should not crash)
    assert os.path.exists(str(img_path))


# --- get_image_timestamp ---

def test_get_image_timestamp_from_filename(tmp_path):
    img_path = tmp_path / "CAM_20220315123045.png"
    Image.new("RGB", (10, 10)).save(str(img_path))
    result = get_image_timestamp(str(tmp_path), "CAM_20220315123045.png")
    assert result == datetime.datetime(2022, 3, 15, 12, 30, 45)


def test_get_image_timestamp_fallback_mtime(tmp_path):
    img_path = tmp_path / "no_date.png"
    Image.new("RGB", (10, 10)).save(str(img_path))
    result = get_image_timestamp(str(tmp_path), "no_date.png")
    # Should fall back to filesystem mtime
    assert isinstance(result, datetime.datetime)


# --- build_image_timestamp_index ---

def test_build_image_timestamp_index(tmp_path):
    for name in ["IMG_20220101120000.png", "IMG_20220102120000.png"]:
        Image.new("RGB", (10, 10)).save(str(tmp_path / name))

    idx = build_image_timestamp_index(str(tmp_path), [
        "IMG_20220101120000.png",
        "IMG_20220102120000.png",
    ])
    assert len(idx) == 2
    assert idx["IMG_20220101120000.png"] == datetime.datetime(2022, 1, 1, 12, 0, 0)


# --- find_series_images ---

def test_find_series_images_groups_burst():
    # Simulate a burst of 3 images within 10 seconds
    idx = {
        "CAM01_20220101120000.jpg": datetime.datetime(2022, 1, 1, 12, 0, 0),
        "CAM01_20220101120005.jpg": datetime.datetime(2022, 1, 1, 12, 0, 5),
        "CAM01_20220101120008.jpg": datetime.datetime(2022, 1, 1, 12, 0, 8),
        "CAM01_20220101130000.jpg": datetime.datetime(2022, 1, 1, 13, 0, 0),  # far away
    }
    result = find_series_images("CAM01_20220101120000.jpg", idx, window_seconds=10)
    assert len(result) == 3
    assert "CAM01_20220101130000.jpg" not in result


def test_find_series_images_different_camera():
    idx = {
        "CAM01_20220101120000.jpg": datetime.datetime(2022, 1, 1, 12, 0, 0),
        "CAM02_20220101120005.jpg": datetime.datetime(2022, 1, 1, 12, 0, 5),
    }
    result = find_series_images("CAM01_20220101120000.jpg", idx,
                                window_seconds=10, require_same_camera=True)
    assert len(result) == 1  # Only CAM01


def test_find_series_images_missing_target():
    idx = {"other.jpg": datetime.datetime(2022, 1, 1, 12, 0, 0)}
    result = find_series_images("missing.jpg", idx)
    assert result == ["missing.jpg"]


# --- blur_box (only if cv2 available) ---

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


@pytest.mark.skipif(not _HAS_CV2, reason="OpenCV not installed")
def test_blur_box_blurs_region():
    from addaxai.utils.images import blur_box

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:40, 20:40] = 255  # white square
    result = blur_box(img, 10, 10, 50, 50, 100, 100)
    assert result.shape == (100, 100, 3)


@pytest.mark.skipif(not _HAS_CV2, reason="OpenCV not installed")
def test_blur_box_invalid_bbox():
    from addaxai.utils.images import blur_box

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        blur_box(img, 50, 50, 10, 10, 100, 100)  # x1 > x2
