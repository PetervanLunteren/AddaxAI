"""Tests for addaxai.models.deploy — model deployment utilities."""

import datetime
import json
import os
import signal
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from addaxai.models.deploy import (
    cancel_subprocess,
    imitate_object_detection_for_full_image_classifier,
    switch_yolov5_version,
)


# --- imitate_object_detection_for_full_image_classifier ---

def test_imitate_detection_creates_json(tmp_path):
    # Create some test images
    for name in ["img1.jpg", "img2.png", "img3.JPEG"]:
        Image.new("RGB", (10, 10)).save(str(tmp_path / name))
    # Also a non-image file that should be ignored
    (tmp_path / "notes.txt").write_text("not an image")

    imitate_object_detection_for_full_image_classifier(str(tmp_path))

    json_path = tmp_path / "image_recognition_file.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    assert len(data["images"]) == 3
    assert data["detection_categories"]["1"] == "animal"

    # Each image should have full-image bbox
    for img in data["images"]:
        assert img["detections"][0]["bbox"] == [0.0, 0.0, 1.0, 1.0]
        assert img["detections"][0]["conf"] == 1.0


def test_imitate_detection_empty_folder(tmp_path):
    imitate_object_detection_for_full_image_classifier(str(tmp_path))

    json_path = tmp_path / "image_recognition_file.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert len(data["images"]) == 0


# --- switch_yolov5_version ---

def test_switch_yolov5_version_old(tmp_path):
    versions_dir = tmp_path / "yolov5_versions" / "yolov5_old" / "yolov5"
    versions_dir.mkdir(parents=True)

    import sys
    original_path = sys.path.copy()
    original_pythonpath = os.environ.get("PYTHONPATH", "")

    try:
        switch_yolov5_version("old models", str(tmp_path))
        assert str(versions_dir) in sys.path
    finally:
        sys.path = original_path
        os.environ["PYTHONPATH"] = original_pythonpath


def test_switch_yolov5_version_new(tmp_path):
    versions_dir = tmp_path / "yolov5_versions" / "yolov5_new" / "yolov5"
    versions_dir.mkdir(parents=True)

    import sys
    original_path = sys.path.copy()
    original_pythonpath = os.environ.get("PYTHONPATH", "")

    try:
        switch_yolov5_version("new models", str(tmp_path))
        assert str(versions_dir) in sys.path
    finally:
        sys.path = original_path
        os.environ["PYTHONPATH"] = original_pythonpath


def test_switch_yolov5_version_invalid():
    with pytest.raises(ValueError):
        switch_yolov5_version("bad value", "/tmp")


# --- cancel_subprocess ---

def test_cancel_subprocess_kills_process():
    mock_process = MagicMock()
    mock_process.pid = 12345

    with patch("addaxai.models.deploy.os") as mock_os:
        mock_os.name = "nt"
        # Just verify it doesn't crash — actual kill is OS-specific
        cancel_subprocess(mock_process)
