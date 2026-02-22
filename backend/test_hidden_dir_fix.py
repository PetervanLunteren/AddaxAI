"""
Tests for the .addaxai frame re-detection fix.

Verifies that:
1. Centralized media_types constants are correct
2. scan_folder_for_images/videos skip hidden directories
3. folder_scanner.scan_folder skips hidden directories
4. megadetector.detect_to_json builds a file-list command (no --recursive)
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── 1. media_types constants ──────────────────────────────────────────────

def test_media_types_are_frozensets():
    from app.core.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

    assert isinstance(IMAGE_EXTENSIONS, frozenset)
    assert isinstance(VIDEO_EXTENSIONS, frozenset)


def test_image_extensions_match_megadetector():
    """IMAGE_EXTENSIONS must match MegaDetector's IMG_EXTENSIONS."""
    from app.core.media_types import IMAGE_EXTENSIONS

    expected = {".jpg", ".jpeg", ".gif", ".png", ".tif", ".tiff", ".bmp"}
    assert IMAGE_EXTENSIONS == expected


def test_video_extensions_superset():
    """VIDEO_EXTENSIONS must include all formats used across the codebase."""
    from app.core.media_types import VIDEO_EXTENSIONS

    # Must include everything previously defined in any file
    must_include = {".mp4", ".avi", ".mpeg", ".mpg", ".mov", ".mkv", ".flv", ".m4v", ".wmv"}
    assert must_include.issubset(VIDEO_EXTENSIONS)


# ── 2. detection_worker scanners ──────────────────────────────────────────

def _make_test_tree(tmp: Path):
    """Create a test directory tree with hidden dirs and media files."""
    # Regular images
    (tmp / "IMG_001.jpg").touch()
    (tmp / "sub").mkdir()
    (tmp / "sub" / "IMG_002.png").touch()

    # Regular video
    (tmp / "VID_001.mp4").touch()

    # Hidden dir with extracted frames (the bug scenario)
    addaxai = tmp / ".addaxai" / "video_frames" / "VID_001.mp4"
    addaxai.mkdir(parents=True)
    (addaxai / "frame000001.jpg").touch()
    (addaxai / "frame000002.jpg").touch()

    # Another hidden dir
    (tmp / ".hidden").mkdir()
    (tmp / ".hidden" / "secret.jpg").touch()


def test_scan_folder_for_images_skips_hidden():
    from app.workers.detection_worker import scan_folder_for_images

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _make_test_tree(tmp_path)

        result = scan_folder_for_images(tmp_path)
        names = [p.name for p in result]

        assert "IMG_001.jpg" in names
        assert "IMG_002.png" in names
        assert len(result) == 2, f"Expected 2 images, got {len(result)}: {names}"

        # Must NOT include frames from .addaxai
        assert "frame000001.jpg" not in names
        assert "frame000002.jpg" not in names
        # Must NOT include files from .hidden
        assert "secret.jpg" not in names


def test_scan_folder_for_videos_skips_hidden():
    from app.workers.detection_worker import scan_folder_for_videos

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _make_test_tree(tmp_path)

        # Also put a video inside .addaxai for good measure
        (tmp_path / ".addaxai" / "clip.mp4").touch()

        result = scan_folder_for_videos(tmp_path)
        names = [p.name for p in result]

        assert "VID_001.mp4" in names
        assert len(result) == 1, f"Expected 1 video, got {len(result)}: {names}"
        assert "clip.mp4" not in names


def test_scan_results_are_sorted():
    from app.workers.detection_worker import scan_folder_for_images

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "c.jpg").touch()
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.jpg").touch()

        result = scan_folder_for_images(tmp_path)
        assert result == sorted(result)


# ── 3. folder_scanner ────────────────────────────────────────────────────

def test_folder_scanner_skips_hidden():
    from app.services.folder_scanner import scan_folder

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _make_test_tree(tmp_path)

        result = scan_folder(str(tmp_path))

        assert result["image_count"] == 2, f"Expected 2 images, got {result['image_count']}"
        assert result["video_count"] == 1, f"Expected 1 video, got {result['video_count']}"
        assert result["total_count"] == 3


# ── 4. megadetector detect_to_json command construction ───────────────────

def test_detect_to_json_uses_file_list_not_recursive():
    """Verify detect_to_json passes a JSON file list and omits --recursive."""
    from app.ml.inference.megadetector import MegaDetectorV1000

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create fake model file
        model_file = tmp_path / "model.pt"
        model_file.touch()

        # Create fake image
        img = tmp_path / "deploy" / "IMG_001.jpg"
        img.parent.mkdir()
        img.touch()

        # Mock environment manager
        env_manager = MagicMock()
        env_manager.get_python.return_value = "/usr/bin/python3"

        detector = MegaDetectorV1000(model_file, env_manager)

        # Capture the subprocess command
        captured_cmd = {}

        def fake_popen(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            proc = MagicMock()
            # stdout must be iterable AND have .close()
            proc.stdout.__iter__ = lambda self: iter([])
            proc.returncode = 0

            # Create a fake output file with valid JSON
            output_path = Path(cmd[-1])
            output_path.write_text(json.dumps({
                "images": [{"file": str(img), "detections": []}],
                "detection_categories": {"1": "animal"},
                "info": {}
            }))
            return proc

        with patch("subprocess.Popen", side_effect=fake_popen):
            result = detector.detect_to_json(
                image_paths=[img],
                deployment_folder=img.parent,
                confidence_threshold=0.1,
            )

        cmd = captured_cmd["cmd"]

        # Must NOT have --recursive or --output_relative_filenames
        assert "--recursive" not in cmd, f"--recursive should not be in command: {cmd}"
        assert "--output_relative_filenames" not in cmd

        # The image_file argument (second-to-last before output) should be a .json file
        image_file_arg = cmd[-2]
        assert image_file_arg.endswith(".json"), f"Expected .json file list, got: {image_file_arg}"


def test_detect_to_json_output_has_relative_paths():
    """Verify post-processing converts absolute paths to relative."""
    from app.ml.inference.megadetector import MegaDetectorV1000

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        model_file = tmp_path / "model.pt"
        model_file.touch()

        deploy = tmp_path / "deploy"
        deploy.mkdir()
        img = deploy / "subdir" / "IMG_001.jpg"
        img.parent.mkdir()
        img.touch()

        env_manager = MagicMock()
        env_manager.get_python.return_value = "/usr/bin/python3"

        detector = MegaDetectorV1000(model_file, env_manager)

        def fake_popen(cmd, **kwargs):
            proc = MagicMock()
            proc.stdout.__iter__ = lambda self: iter([])
            proc.returncode = 0

            output_path = Path(cmd[-1])
            output_path.write_text(json.dumps({
                "images": [{"file": str(img), "detections": []}],
                "detection_categories": {"1": "animal"},
                "info": {}
            }))
            return proc

        with patch("subprocess.Popen", side_effect=fake_popen):
            result_path = detector.detect_to_json(
                image_paths=[img],
                deployment_folder=deploy,
                confidence_threshold=0.1,
            )

        # Read the output and verify paths are relative
        with open(result_path) as f:
            output = json.load(f)

        file_path = output["images"][0]["file"]
        assert file_path == "subdir/IMG_001.jpg", f"Expected relative path, got: {file_path}"
        assert not Path(file_path).is_absolute(), "Path should be relative"
