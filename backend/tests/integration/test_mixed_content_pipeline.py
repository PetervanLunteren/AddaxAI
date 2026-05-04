"""
Integration tests: video+image merge and frame linkage in DB.

Tests merge_json_files() for classification ID unification and
load_json_to_database() for video/frame File record creation.
"""

import json
from unittest.mock import patch

from app.ml.json_pipeline import load_json_to_database, merge_json_files
from app.models import Detection, File

from .conftest import (
    build_detection_json,
    create_video_frames,
    write_json,
)


def test_merge_unifies_classification_ids(deployment_scaffold):
    """Same label from video JSON (id='1') and image JSON (id='2') gets unified ID."""
    s = deployment_scaffold
    artifacts = s["artifacts"]

    video_json = build_detection_json(
        images=[{
            "file": "videos/clip.mp4",
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
                 "classifications": [[1, 0.8], [2, 0.2]], "frame_number": 0},
            ],
        }],
        classification_categories={"1": "zebra", "2": "giraffe"},
    )
    image_json = build_detection_json(
        images=[{
            "file": "subdir/img_001.jpg",
            "detections": [
                {"category": "1", "conf": 0.85, "bbox": [0.2, 0.3, 0.4, 0.5],
                 "classifications": [[1, 0.7], [2, 0.3]]},
            ],
        }],
        # Different ID mapping: "1"=giraffe, "2"=zebra (swapped vs video)
        classification_categories={"1": "giraffe", "2": "zebra"},
    )

    vid_json_path = write_json(artifacts / "video_results.json", video_json)
    img_json_path = write_json(artifacts / "image_results.json", image_json)
    merged_path = artifacts / "merged.json"

    merge_json_files(
        [vid_json_path, img_json_path],
        merged_path,
        s["deployment"].id,
    )

    with open(merged_path) as f:
        merged = json.load(f)

    # Unified categories should have both labels with consistent IDs
    cats = merged["classification_categories"]
    assert "zebra" in cats.values()
    assert "giraffe" in cats.values()

    # All classification IDs should reference the unified mapping
    for img in merged["images"]:
        for det in img.get("detections", []):
            for cls_id, _ in det.get("classifications", []):
                assert str(cls_id) in cats, f"ID {cls_id} not in unified categories"


def test_load_creates_video_image_and_frame_files(deployment_scaffold):
    """file_type='video', 'image', 'frame' records; frame has source_video_id FK."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    # Create video frames on disk
    create_video_frames(s["artifacts"], "videos/clip.mp4", [0, 30, 60])

    images = [
        {
            "file": "videos/clip.mp4",
            "best_frame_number": 30,
            "frame_rate": 30.0,
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
                 "frame_number": 0},
                {"category": "1", "conf": 0.85, "bbox": [0.2, 0.3, 0.4, 0.5],
                 "frame_number": 30},
            ],
        },
        {
            "file": "subdir/img_001.jpg",
            "detections": [
                {"category": "1", "conf": 0.7, "bbox": [0.1, 0.1, 0.2, 0.2]},
            ],
        },
    ]
    md_json = build_detection_json(images)
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    all_files = db.query(File).filter(File.deployment_id == s["deployment"].id).all()
    types = {f.file_type for f in all_files}
    assert types == {"video", "image", "frame"}

    video_files = [f for f in all_files if f.file_type == "video"]
    assert len(video_files) == 1
    assert video_files[0].best_frame_number == 30

    frame_files = [f for f in all_files if f.file_type == "frame"]
    assert len(frame_files) == 3  # 3 extracted frames
    for ff in frame_files:
        assert ff.source_video_id == video_files[0].id

    image_files = [f for f in all_files if f.file_type == "image"]
    assert len(image_files) == 1


def test_detections_linked_to_frames_not_videos(deployment_scaffold):
    """Video detections with frame_number → detection.file_id points to frame File."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    create_video_frames(s["artifacts"], "videos/clip.mp4", [0, 30])

    images = [{
        "file": "videos/clip.mp4",
        "frame_rate": 30.0,
        "detections": [
            {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
             "frame_number": 0},
            {"category": "1", "conf": 0.85, "bbox": [0.2, 0.3, 0.4, 0.5],
             "frame_number": 30},
        ],
    }]
    md_json = build_detection_json(images)
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    video_file = (
        db.query(File)
        .filter(File.deployment_id == s["deployment"].id, File.file_type == "video")
        .one()
    )

    detections = db.query(Detection).all()
    assert len(detections) == 2

    for det in detections:
        # Detection should be linked to a frame, not the video
        linked_file = db.query(File).filter(File.id == det.file_id).one()
        assert linked_file.file_type == "frame"
        assert linked_file.source_video_id == video_file.id


def test_blank_video_creates_blank_frames(deployment_scaffold):
    """Video with no detections → all frames have observation_type='blank'."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    create_video_frames(s["artifacts"], "videos/clip.mp4", [0, 30])

    images = [{
        "file": "videos/clip.mp4",
        "frame_rate": 30.0,
        "detections": [],
    }]
    md_json = build_detection_json(images)
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    frame_files = (
        db.query(File)
        .filter(File.deployment_id == s["deployment"].id, File.file_type == "frame")
        .all()
    )
    assert len(frame_files) == 2
    for ff in frame_files:
        assert ff.observation_type == "blank"

    video_file = (
        db.query(File)
        .filter(File.deployment_id == s["deployment"].id, File.file_type == "video")
        .one()
    )
    assert video_file.observation_type == "blank"
