"""
Integration tests: a box the classifier calls "nothing here" is kept.

When a model's top answer for a box is a non-label class (blank, empty,
false detection, ...), the box is stored like any other detection, with
that label and the model's score. It is the same row a person creates by
pressing X on the Labels page, minus the verified flag, so every count
already ignores it (`is_a_real_detection()`) and the Labels grid shows
it as a card to confirm or rescue.

Until 2026-09 such boxes were dropped at ingest. That deleted the model's
own evidence with no trace: a 0.83 MegaDetector box that ARC-ADS called
"false detection" at 95% was in the recognition file when no classifier
ran and gone when one did (Dan Morris, 2026-09-28). On this machine one
run had dropped 11,667 of 30,810 boxes that way.
"""

from unittest.mock import patch

from app.ml.json_pipeline import load_json_to_database
from app.models import Detection, File

from .conftest import build_detection_json, write_json


def _load(s: dict, images: list[dict], categories: dict[str, str] | None):
    md_json = build_detection_json(images, classification_categories=categories)
    json_path = write_json(s["artifacts"] / "results.json", md_json)
    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        return load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=s["deploy_dir"],
            job_id=s["job"].id,
            db=s["db"],
            artifacts_folder=s["artifacts"],
        )


def _rel(s: dict, index: int) -> str:
    return str(s["img_paths"][index].relative_to(s["deploy_dir"]))


def test_a_box_the_model_rejected_is_stored_with_its_label(deployment_scaffold):
    """The row exists, carries the model's verdict and score, and is
    unverified: a person has not looked at it yet."""
    s = deployment_scaffold
    result = _load(
        s,
        [{
            "file": _rel(s, 0),
            "detections": [{
                "category": "1",
                "conf": 0.8,
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "classifications": [["1", 0.97], ["2", 0.02]],
            }],
        }],
        {"1": "blank", "2": "lion"},
    )

    assert result.total_detections == 1
    assert result.animal_detections == 1
    assert result.classified_detections == 1
    det = s["db"].query(Detection).one()
    assert det.label == "blank"
    assert det.label_confidence == 0.97
    assert det.confidence == 0.8
    assert det.verified is False
    assert det.classification_method == "machine"


def test_a_file_with_only_rejected_boxes_reads_blank(deployment_scaffold):
    """The row is kept, but it is not an observation: the file's subject
    is decided by `strongest_passing_detection`, which skips it."""
    s = deployment_scaffold
    _load(
        s,
        [{
            "file": _rel(s, 0),
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
        }],
        {"1": "blank"},
    )

    assert s["db"].query(Detection).count() == 2
    f = s["db"].query(File).filter(File.deployment_id == s["deployment"].id).one()
    assert f.observation_type == "blank"


def test_a_real_box_beside_a_rejected_one_names_the_file(deployment_scaffold):
    """Blank, lion and a person on one file: three rows, and the file is
    an animal because the lion is the strongest real box."""
    s = deployment_scaffold
    result = _load(
        s,
        [{
            "file": _rel(s, 0),
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
                {"category": "2", "conf": 0.7, "bbox": [0.6, 0.6, 0.2, 0.2]},
            ],
        }],
        {"1": "blank", "2": "lion"},
    )

    assert result.total_detections == 3
    assert result.animal_detections == 2
    assert result.person_detections == 1
    labels = {d.label for d in s["db"].query(Detection).all()}
    assert labels == {"blank", "lion", None}
    f = s["db"].query(File).filter(File.deployment_id == s["deployment"].id).one()
    assert f.observation_type == "animal"


def test_an_unclassified_animal_is_stored_without_a_label(deployment_scaffold):
    """No classifier output is not a rejection: label stays NULL."""
    s = deployment_scaffold
    result = _load(
        s,
        [{
            "file": _rel(s, 0),
            "detections": [
                {"category": "1", "conf": 0.7, "bbox": [0.1, 0.2, 0.3, 0.4]},
            ],
        }],
        None,
    )

    assert result.total_detections == 1
    det = s["db"].query(Detection).one()
    assert det.label is None
    assert det.category == "animal"
