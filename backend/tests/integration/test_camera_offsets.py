"""
Integration tests: per-camera clock offsets at ingest.

A paired deployment carries `camera_offsets`, seconds per camera subfolder,
added on top of the whole-deployment `datetime_offset_seconds`. The ingest
bakes both into `File.captured_at_local`. Root-level files get only the base.
"""

from datetime import datetime
from unittest.mock import patch

from app.ml.json_pipeline import load_json_to_database
from app.models import File

from .conftest import build_detection_json, create_tiny_jpeg, write_json

EXIF = {"DateTimeOriginal": "2024:06:15 10:00:00"}
RAW = datetime(2024, 6, 15, 10, 0, 0)


def _entry(rel: str) -> dict:
    return {
        "file": rel,
        "exif_metadata": EXIF,
        "detections": [{"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4]}],
    }


def _load(s, images, **kwargs):
    json_path = write_json(s["artifacts"] / "results.json", build_detection_json(images))
    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        return load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=s["deploy_dir"],
            job_id=s["job"].id,
            db=s["db"],
            artifacts_folder=s["artifacts"],
            **kwargs,
        )


def _captured(s, path) -> datetime:
    return s["db"].query(File).filter(File.file_path == str(path)).one().captured_at_local


def test_camera_offset_adds_to_the_base_for_that_subfolder_only(deployment_scaffold):
    s = deployment_scaffold
    deploy_dir = s["deploy_dir"]
    (deploy_dir / "cam_b").mkdir()
    cam_a = s["img_paths"][0]  # lives in subdir/
    cam_b = create_tiny_jpeg(deploy_dir / "cam_b" / "img.jpg")
    root = create_tiny_jpeg(deploy_dir / "root.jpg")

    _load(
        s,
        [_entry(str(p.relative_to(deploy_dir))) for p in (cam_a, cam_b, root)],
        datetime_offset_seconds=-3600,
        camera_offsets={"cam_b": 60},
    )

    assert _captured(s, cam_a) == datetime(2024, 6, 15, 9, 0, 0)
    assert _captured(s, cam_b) == datetime(2024, 6, 15, 9, 1, 0)
    assert _captured(s, root) == datetime(2024, 6, 15, 9, 0, 0)


def test_camera_offset_alone_without_a_base(deployment_scaffold):
    s = deployment_scaffold
    cam_a = s["img_paths"][0]
    rel = str(cam_a.relative_to(s["deploy_dir"]))

    _load(s, [_entry(rel)], camera_offsets={"subdir": -90})

    assert _captured(s, cam_a) == datetime(2024, 6, 15, 9, 58, 30)


def test_no_camera_offsets_leaves_the_base_behaviour(deployment_scaffold):
    s = deployment_scaffold
    cam_a = s["img_paths"][0]
    rel = str(cam_a.relative_to(s["deploy_dir"]))

    _load(s, [_entry(rel)], datetime_offset_seconds=30, camera_offsets={})

    assert _captured(s, cam_a) == datetime(2024, 6, 15, 10, 0, 30)
