"""
Integration tests: images-only pipeline (JSON → DB records).

Calls load_json_to_database() directly with fixture JSON and real tiny
files on disk. No mocks needed — this function only reads JSON + file stats.
"""

from unittest.mock import patch

from app.ml.json_pipeline import load_json_to_database
from app.models import Detection, File

from .conftest import build_detection_json, write_json


def test_load_creates_file_records(deployment_scaffold):
    """3 image files → 3 File records with correct type, path, observation_type."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    images = []
    for p in s["img_paths"]:
        rel = str(p.relative_to(deploy_dir))
        images.append({
            "file": rel,
            "detections": [
                {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4]},
            ],
        })

    md_json = build_detection_json(images)
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        result = load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    assert result.total_files == 3

    files = db.query(File).filter(File.deployment_id == s["deployment"].id).all()
    assert len(files) == 3

    for f in files:
        assert f.file_type == "image"
        assert f.file_format == "jpg"
        assert f.observation_type == "animal"
        assert deploy_dir.name in f.file_path  # absolute path contains deploy dir


def test_load_creates_detection_records(deployment_scaffold):
    """Correct detection count, category mapping, bbox, label, classification_method."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    rel = str(s["img_paths"][0].relative_to(deploy_dir))
    images = [{
        "file": rel,
        "detections": [
            {
                "category": "1",
                "conf": 0.95,
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "classifications": [[1, 0.85], [2, 0.15]],
            },
        ],
    }]
    md_json = build_detection_json(
        images,
        classification_categories={"1": "lion", "2": "zebra"},
    )
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        result = load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    assert result.total_detections == 1
    assert result.animal_detections == 1
    assert result.classified_detections == 1

    det = db.query(Detection).one()
    assert det.category == "animal"
    assert det.confidence == 0.95
    assert det.bbox_x == 0.1
    assert det.bbox_y == 0.2
    assert det.bbox_width == 0.3
    assert det.bbox_height == 0.4
    assert det.label == "lion"
    assert det.label_confidence == 0.85
    assert det.classification_method == "machine"


def test_load_stores_raw_labels(deployment_scaffold):
    """Phase 6 stores raw top-1 without exclusion (exclusion is Phase 7)."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    rel = str(s["img_paths"][0].relative_to(deploy_dir))
    images = [{
        "file": rel,
        "detections": [
            {
                "category": "1",
                "conf": 0.9,
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "classifications": [[1, 0.6], [2, 0.3], [3, 0.1]],
            },
        ],
    }]
    md_json = build_detection_json(
        images,
        classification_categories={"1": "lion", "2": "zebra", "3": "giraffe"},
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

    det = db.query(Detection).one()
    # Raw top-1 is lion (exclusion happens in Phase 7, not here)
    assert det.label == "lion"
    assert abs(det.label_confidence - 0.6) < 0.01


def test_observation_type_priority(deployment_scaffold):
    """File with both animal + person detections → observation_type='animal'."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    rel = str(s["img_paths"][0].relative_to(deploy_dir))
    images = [{
        "file": rel,
        "detections": [
            {"category": "2", "conf": 0.95, "bbox": [0.0, 0.0, 0.5, 0.5]},
            {"category": "1", "conf": 0.80, "bbox": [0.5, 0.5, 0.3, 0.3]},
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

    f = db.query(File).filter(File.deployment_id == s["deployment"].id).one()
    assert f.observation_type == "animal"
