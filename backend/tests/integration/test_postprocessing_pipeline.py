"""
Integration tests: postprocessing pipeline (smoothing, exclusion, reload).

Tests update_database_from_smoothed_results(), run_postprocessing_for_deployment(),
reload_raw_classifications_from_json(), and build_sequence_information().

Mocks: subprocess.run, _get_ml_python_path, _find_classification_model_dir
(to avoid needing the ML conda env and model files).
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.ml.json_pipeline import load_json_to_database
from app.ml.postprocessing import (
    build_sequence_information,
    reload_raw_classifications_from_json,
    run_postprocessing_for_deployment,
    update_database_from_smoothed_results,
)
from app.models import Detection, File
from tests.conftest import make_file

from .conftest import build_detection_json, create_tiny_jpeg, create_video_frames, write_json


def _load_basic_images(
    s: dict, label_map: dict[str, str] | None = None
) -> Path:
    """Load 3 images with animal detections into DB, return json_path."""
    db, deploy_dir = s["db"], s["deploy_dir"]
    classification_categories = label_map or {"1": "lion", "2": "zebra", "3": "giraffe"}

    images = []
    for i, p in enumerate(s["img_paths"]):
        rel = str(p.relative_to(deploy_dir))
        images.append({
            "file": rel,
            "exif_metadata": {
                "DateTimeOriginal": (
                    datetime(2024, 6, 15, 10, 0, 0) + timedelta(minutes=i)
                ).strftime("%Y:%m:%d %H:%M:%S"),
            },
            "detections": [
                {
                    "category": "1",
                    "conf": 0.9,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "classifications": [[1, 0.7], [2, 0.2], [3, 0.1]],
                },
            ],
        })

    md_json = build_detection_json(images, classification_categories=classification_categories)
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

    return json_path


def test_update_db_from_smoothed_results(deployment_scaffold):
    """Label/confidence updated in DB; correct {updated, unchanged, errors} counts."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    _load_basic_images(s)

    # Verify initial state
    dets = db.query(Detection).all()
    assert len(dets) == 3
    assert all(d.label == "lion" for d in dets)

    # Build smoothed results that change label for first 2 images
    smoothed_images = []
    files = (
        db.query(File)
        .filter(File.deployment_id == s["deployment"].id)
        .order_by(File.timestamp.asc())
        .all()
    )

    for i, f in enumerate(files):
        rel = str(Path(f.file_path).relative_to(deploy_dir))

        if i < 2:
            # Change to zebra
            cls = [[2, 0.8], [1, 0.2]]
        else:
            # Keep as lion (unchanged)
            cls = [[1, 0.7], [2, 0.2], [3, 0.1]]

        smoothed_images.append({
            "file": rel,
            "detections": [
                {
                    "category": "1",
                    "conf": 0.9,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "classifications": cls,
                },
            ],
        })

    smoothed = build_detection_json(
        smoothed_images,
        classification_categories={"1": "lion", "2": "zebra", "3": "giraffe"},
    )

    counts = update_database_from_smoothed_results(
        deployment_id=s["deployment"].id,
        smoothed_results=smoothed,
        deployment_folder=deploy_dir,
        db=db,
    )

    assert counts["updated"] == 2
    assert counts["unchanged"] == 1
    assert counts["errors"] == 0

    # Verify DB was updated
    updated_dets = db.query(Detection).join(File).order_by(File.timestamp.asc()).all()
    assert updated_dets[0].label == "zebra"
    assert updated_dets[1].label == "zebra"
    assert updated_dets[2].label == "lion"


def test_smoothing_matches_by_bbox_and_frame(deployment_scaffold):
    """Matching works for images (path+bbox) and video frames (path+bbox+frame_number)."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    create_video_frames(s["artifacts"], "videos/clip.mp4", [0, 30])

    images = [
        {
            "file": "videos/clip.mp4",
            "frame_rate": 30.0,
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
                 "frame_number": 0, "classifications": [[1, 0.8]]},
                {"category": "1", "conf": 0.85, "bbox": [0.5, 0.5, 0.2, 0.2],
                 "frame_number": 30, "classifications": [[1, 0.7]]},
            ],
        },
        {
            "file": "subdir/img_001.jpg",
            "detections": [
                {"category": "1", "conf": 0.75, "bbox": [0.2, 0.3, 0.4, 0.5],
                 "classifications": [[1, 0.6]]},
            ],
        },
    ]
    md_json = build_detection_json(
        images, classification_categories={"1": "lion", "2": "zebra"}
    )
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

    # Build smoothed results: change all to zebra
    smoothed_images = [
        {
            "file": "videos/clip.mp4",
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
                 "frame_number": 0, "classifications": [[2, 0.9]]},
                {"category": "1", "conf": 0.85, "bbox": [0.5, 0.5, 0.2, 0.2],
                 "frame_number": 30, "classifications": [[2, 0.85]]},
            ],
        },
        {
            "file": "subdir/img_001.jpg",
            "detections": [
                {"category": "1", "conf": 0.75, "bbox": [0.2, 0.3, 0.4, 0.5],
                 "classifications": [[2, 0.8]]},
            ],
        },
    ]
    smoothed = build_detection_json(
        smoothed_images, classification_categories={"1": "lion", "2": "zebra"}
    )

    counts = update_database_from_smoothed_results(
        deployment_id=s["deployment"].id,
        smoothed_results=smoothed,
        deployment_folder=deploy_dir,
        db=db,
    )

    assert counts["updated"] == 3
    assert counts["errors"] == 0

    for det in db.query(Detection).all():
        assert det.label == "zebra"


def test_label_exclusion_applied_before_smoothing(deployment_scaffold):
    """JSON passed to smoothing subprocess has excluded labels filtered out."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    _load_basic_images(s)

    # Set up project with exclusion
    s["project"].excluded_classes = ["lion"]
    s["project"].event_smoothing = True
    s["project"].taxonomic_rollup = False
    s["project"].independence_interval = 1800
    s["project"].detection_threshold = 0.5
    db.flush()

    json_path = s["artifacts"] / "results.json"

    captured_input = {}

    def fake_subprocess_run(cmd, **kwargs):
        """Capture the input JSON passed to subprocess and write output."""
        input_path = cmd[2]  # [python, script, input, opts, output]
        output_path = cmd[4]

        with open(input_path) as f:
            captured_input["data"] = json.load(f)

        # Write input as output (passthrough)
        with open(output_path, "w") as f:
            json.dump(captured_input["data"], f)

        class FakeResult:
            returncode = 0
            stderr = ""

        return FakeResult()

    with (
        patch("app.ml.postprocessing.subprocess.run", side_effect=fake_subprocess_run),
        patch("app.ml.postprocessing._get_ml_python_path", return_value="/fake/python"),
        patch("app.ml.postprocessing._find_classification_model_dir", return_value=None),
    ):
        run_postprocessing_for_deployment(
            deployment_id=s["deployment"].id,
            json_path=json_path,
            deployment_folder=deploy_dir,
            project=s["project"],
            db=db,
        )

    # Verify lion was excluded from the JSON passed to subprocess
    data = captured_input["data"]
    for img in data["images"]:
        for det in img.get("detections", []):
            for cls_id, _ in det.get("classifications", []):
                label_name = data["classification_categories"].get(str(cls_id))
                assert label_name != "lion", "Excluded label should be filtered out"


def test_reload_raw_classifications(deployment_scaffold):
    """After smoothing, reload_raw_classifications_from_json() reverts to raw values."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    json_path = _load_basic_images(s)

    # First verify initial label
    dets = db.query(Detection).all()
    assert all(d.label == "lion" for d in dets)

    # Simulate smoothing: change all to zebra
    files = db.query(File).filter(File.deployment_id == s["deployment"].id).all()
    smoothed_images = []
    for f in files:
        rel = str(Path(f.file_path).relative_to(deploy_dir))
        smoothed_images.append({
            "file": rel,
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
                 "classifications": [[2, 0.9]]},
            ],
        })

    smoothed = build_detection_json(
        smoothed_images,
        classification_categories={"1": "lion", "2": "zebra", "3": "giraffe"},
    )
    update_database_from_smoothed_results(
        s["deployment"].id, smoothed, deploy_dir, db
    )

    # Verify smoothing took effect
    dets = db.query(Detection).all()
    assert all(d.label == "zebra" for d in dets)

    # Now reload raw → should revert to lion
    counts = reload_raw_classifications_from_json(
        deployment_id=s["deployment"].id,
        json_path=json_path,
        deployment_folder=deploy_dir,
        db=db,
    )

    assert counts["updated"] == 3
    dets = db.query(Detection).all()
    assert all(d.label == "lion" for d in dets)


def test_verified_detections_skipped_during_reprocessing(deployment_scaffold):
    """Verified detections keep their labels when smoothed results are applied."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    _load_basic_images(s)

    # Verify initial state: all lion
    dets = db.query(Detection).join(File).order_by(File.timestamp.asc()).all()
    assert len(dets) == 3
    assert all(d.label == "lion" for d in dets)

    # Mark first detection as verified
    dets[0].verified = True
    dets[0].verified_at = datetime.utcnow()
    db.flush()

    # Build smoothed results changing all 3 to zebra
    files = (
        db.query(File)
        .filter(File.deployment_id == s["deployment"].id)
        .order_by(File.timestamp.asc())
        .all()
    )
    smoothed_images = []
    for f in files:
        rel = str(Path(f.file_path).relative_to(deploy_dir))
        smoothed_images.append({
            "file": rel,
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
                 "classifications": [[2, 0.9]]},
            ],
        })

    smoothed = build_detection_json(
        smoothed_images,
        classification_categories={"1": "lion", "2": "zebra", "3": "giraffe"},
    )

    counts = update_database_from_smoothed_results(
        deployment_id=s["deployment"].id,
        smoothed_results=smoothed,
        deployment_folder=deploy_dir,
        db=db,
    )

    assert counts["skipped_verified"] == 1
    assert counts["updated"] == 2

    # Verified detection keeps original label; others updated
    updated_dets = db.query(Detection).join(File).order_by(File.timestamp.asc()).all()
    assert updated_dets[0].label == "lion"
    assert updated_dets[1].label == "zebra"
    assert updated_dets[2].label == "zebra"


def test_verified_detections_skipped_during_raw_reload(deployment_scaffold):
    """Verified detections survive reload_raw_classifications_from_json()."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]
    json_path = _load_basic_images(s)

    # Smoothing: change all to zebra
    files = db.query(File).filter(File.deployment_id == s["deployment"].id).all()
    smoothed_images = []
    for f in files:
        rel = str(Path(f.file_path).relative_to(deploy_dir))
        smoothed_images.append({
            "file": rel,
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4],
                 "classifications": [[2, 0.9]]},
            ],
        })

    smoothed = build_detection_json(
        smoothed_images,
        classification_categories={"1": "lion", "2": "zebra", "3": "giraffe"},
    )
    update_database_from_smoothed_results(
        s["deployment"].id, smoothed, deploy_dir, db
    )

    # Verify all are now zebra
    dets = db.query(Detection).join(File).order_by(File.timestamp.asc()).all()
    assert all(d.label == "zebra" for d in dets)

    # Mark first detection as verified (while labeled "zebra")
    dets[0].verified = True
    dets[0].verified_at = datetime.utcnow()
    db.flush()

    # Reload raw → would revert to lion, but verified detection should be protected
    counts = reload_raw_classifications_from_json(
        deployment_id=s["deployment"].id,
        json_path=json_path,
        deployment_folder=deploy_dir,
        db=db,
    )

    assert counts["skipped_verified"] == 1

    reloaded_dets = db.query(Detection).join(File).order_by(File.timestamp.asc()).all()
    assert reloaded_dets[0].label == "zebra"  # protected
    assert reloaded_dets[1].label == "lion"   # reverted
    assert reloaded_dets[2].label == "lion"   # reverted


def test_build_sequence_groups_by_interval(deployment_scaffold):
    """Files within interval → same seq_id; gap > interval → different seq_id."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    base_time = datetime(2024, 6, 15, 10, 0, 0)

    # Create 4 files: 3 close together, then a gap
    timestamps = [
        base_time,
        base_time + timedelta(minutes=5),
        base_time + timedelta(minutes=10),
        base_time + timedelta(hours=2),  # big gap
    ]

    for i, ts in enumerate(timestamps):
        p = create_tiny_jpeg(deploy_dir / f"seq_img_{i}.jpg")
        make_file(
            db,
            deployment_id=s["deployment"].id,
            file_path=str(p),
            file_type="image",
            timestamp=ts,
        )
    db.flush()

    seq_info = build_sequence_information(
        deployment_id=s["deployment"].id,
        independence_interval=1800,  # 30 min
        db=db,
    )

    assert len(seq_info) == 4
    # First 3 should share a seq_id, 4th should differ
    assert seq_info[0]["seq_id"] == seq_info[1]["seq_id"]
    assert seq_info[1]["seq_id"] == seq_info[2]["seq_id"]
    assert seq_info[2]["seq_id"] != seq_info[3]["seq_id"]
