"""
Integration tests: non-label detection skip during DB loading.

Detections classified exclusively as non-label classes (blank, empty,
false detection, etc.) are not loaded to the database. This filters
MegaDetector false positives that the classifier identified as blank.
"""

from unittest.mock import patch

from app.ml.json_pipeline import load_json_to_database
from app.models import Detection, File

from .conftest import build_detection_json, write_json


def test_skip_blank_classified_detection(deployment_scaffold):
    """Animal detection classified as 'blank' is not loaded to DB."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    rel = str(s["img_paths"][0].relative_to(deploy_dir))
    images = [{
        "file": rel,
        "detections": [{
            "category": "1",
            "conf": 0.8,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "classifications": [["1", 0.97]],
        }],
    }]
    md_json = build_detection_json(
        images,
        classification_categories={"1": "blank"},
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

    assert result.total_detections == 0
    assert result.animal_detections == 0
    assert db.query(Detection).count() == 0


def test_keep_unclassified_animal(deployment_scaffold):
    """Animal detection without classifications is loaded (not skipped)."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    rel = str(s["img_paths"][0].relative_to(deploy_dir))
    images = [{
        "file": rel,
        "detections": [{
            "category": "1",
            "conf": 0.7,
            "bbox": [0.1, 0.2, 0.3, 0.4],
        }],
    }]
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

    assert result.total_detections == 1
    det = db.query(Detection).one()
    assert det.label is None
    assert det.category == "animal"


def test_keep_person_and_vehicle(deployment_scaffold):
    """Person and vehicle detections are never skipped."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    rel = str(s["img_paths"][0].relative_to(deploy_dir))
    images = [{
        "file": rel,
        "detections": [
            {"category": "2", "conf": 0.9, "bbox": [0.0, 0.0, 0.5, 0.5]},
            {"category": "3", "conf": 0.8, "bbox": [0.5, 0.5, 0.3, 0.3]},
        ],
    }]
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

    assert result.total_detections == 2
    assert result.person_detections == 1
    assert result.vehicle_detections == 1


def test_observation_type_blank_when_all_skipped(deployment_scaffold):
    """File with only blank-classified animals gets observation_type='blank'."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    rel = str(s["img_paths"][0].relative_to(deploy_dir))
    images = [{
        "file": rel,
        "detections": [
            {
                "category": "1",
                "conf": 0.8,
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "classifications": [["1", 0.95]],
            },
            {
                "category": "1",
                "conf": 0.6,
                "bbox": [0.5, 0.5, 0.2, 0.2],
                "classifications": [["1", 0.99]],
            },
        ],
    }]
    md_json = build_detection_json(
        images,
        classification_categories={"1": "blank"},
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

    f = db.query(File).filter(
        File.deployment_id == s["deployment"].id
    ).one()
    assert f.observation_type == "blank"


def test_mixed_detections_skip_only_non_label(deployment_scaffold):
    """Only blank-classified animal is skipped; lion animal and person kept."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    rel = str(s["img_paths"][0].relative_to(deploy_dir))
    images = [{
        "file": rel,
        "detections": [
            {
                "category": "1",
                "conf": 0.8,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "classifications": [["1", 0.95]],
            },
            {
                "category": "1",
                "conf": 0.9,
                "bbox": [0.3, 0.3, 0.2, 0.2],
                "classifications": [["2", 0.85]],
            },
            {
                "category": "2",
                "conf": 0.7,
                "bbox": [0.6, 0.6, 0.2, 0.2],
            },
        ],
    }]
    md_json = build_detection_json(
        images,
        classification_categories={"1": "blank", "2": "lion"},
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

    assert result.total_detections == 2
    assert result.animal_detections == 1
    assert result.person_detections == 1

    dets = db.query(Detection).all()
    assert len(dets) == 2
    labels = {d.label for d in dets}
    assert "lion" in labels

    f = db.query(File).filter(
        File.deployment_id == s["deployment"].id
    ).one()
    assert f.observation_type == "animal"


def test_statistics_exclude_skipped(deployment_scaffold):
    """Pipeline result counts exclude skipped non-label detections."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    # Use two images to get multiple detections
    images = []
    for img_path in s["img_paths"][:1]:
        rel = str(img_path.relative_to(deploy_dir))
        images.append({
            "file": rel,
            "detections": [
                {
                    "category": "1",
                    "conf": 0.9,
                    "bbox": [0.0, 0.0, 0.3, 0.3],
                    "classifications": [["1", 0.95]],
                },
                {
                    "category": "1",
                    "conf": 0.8,
                    "bbox": [0.3, 0.3, 0.3, 0.3],
                    "classifications": [["2", 0.90]],
                },
                {
                    "category": "1",
                    "conf": 0.7,
                    "bbox": [0.6, 0.6, 0.2, 0.2],
                    "classifications": [["3", 0.80]],
                },
            ],
        })

    md_json = build_detection_json(
        images,
        classification_categories={
            "1": "blank",
            "2": "empty",
            "3": "lion",
        },
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
