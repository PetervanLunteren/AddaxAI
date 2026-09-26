"""Detector output reuse must preserve class IDs and confidence verbatim."""

import json

import pytest

from app.ml.detector_classification import (
    add_detector_classes_as_classifications,
    reuse_detector_classes_in_json,
)
from app.ml.schemas.model_manifest import uses_detection_classes_for_classification


@pytest.mark.parametrize(
    "class_names",
    [
        {"0": "fox", "1": "Fox "},
        {"00": "fox", "1": "deer"},
    ],
)
def test_detector_alias_requires_unique_names_and_canonical_ids(class_names):
    from types import SimpleNamespace

    manifest = SimpleNamespace(
        model_category="detection",
        managed=True,
        local_only=True,
        detector_backend="rtdetrv2",
        class_names=class_names,
    )

    assert not uses_detection_classes_for_classification(manifest)


@pytest.mark.parametrize(
    ("explicit_alias", "expected"),
    [(False, False), (True, True), (None, True)],
)
def test_manifest_alias_flag_controls_new_packs_and_preserves_legacy(
    explicit_alias, expected
):
    from types import SimpleNamespace

    manifest = SimpleNamespace(
        model_category="detection",
        managed=True,
        local_only=True,
        detector_backend="rtdetrv2",
        class_names={"0": "animal", "1": "person", "2": "vehicle"},
        classification_uses_detection_classes=explicit_alias,
    )

    assert uses_detection_classes_for_classification(manifest) is expected


def test_reuses_zero_based_detector_classes_and_confidence_without_inference():
    document = {
        "detection_categories": {"0": "fox", "1": "deer"},
        "images": [
            {
                "file": "one.jpg",
                "detections": [
                    {"category": "0", "conf": 0.73125, "bbox": [0, 0, 1, 1]},
                    {"category": 1, "conf": 0.25, "bbox": [0.1, 0.2, 0.3, 0.4]},
                ],
            }
        ],
    }

    result = add_detector_classes_as_classifications(document)

    assert result["classification_categories"] == {"0": "fox", "1": "deer"}
    assert result["images"][0]["detections"][0]["classifications"] == [["0", 0.73125]]
    assert result["images"][0]["detections"][1]["classifications"] == [["1", 0.25]]


@pytest.mark.parametrize("detections", [[], None])
def test_empty_and_null_detection_lists_are_valid(detections):
    document = {
        "detection_categories": {"0": "fox"},
        "images": [{"file": "empty.jpg", "detections": detections}],
    }

    result = add_detector_classes_as_classifications(document)

    assert result["classification_categories"] == {"0": "fox"}
    assert result["images"][0]["detections"] is detections


@pytest.mark.parametrize(
    "document, message",
    [
        ({"images": []}, "detection_categories"),
        ({"detection_categories": {"0": "fox"}, "images": None}, "images list"),
        (
            {
                "detection_categories": {"0": "fox"},
                "images": [{"detections": [{"category": "1", "conf": 0.5}]}],
            },
            "no declared class name",
        ),
        (
            {
                "detection_categories": {"0": "fox"},
                "images": [{"detections": [{"category": "0", "conf": 1.01}]}],
            },
            "confidence",
        ),
    ],
)
def test_rejects_undeclared_classes_and_invalid_detector_confidence(document, message):
    with pytest.raises(ValueError, match=message):
        add_detector_classes_as_classifications(document)


def test_reuse_updates_image_and_video_json_atomically(tmp_path):
    path = tmp_path / "detection.json"
    path.write_text(
        json.dumps(
            {
                "detection_categories": {"0": "fox"},
                "images": [
                    {"file": "frame.jpg", "detections": [{"category": "0", "conf": 0.8}]},
                    {"file": "failed.mp4", "detections": None},
                ],
            }
        ),
        encoding="utf-8",
    )

    reuse_detector_classes_in_json(path)
    # Reapply to a completed cached JSON, as the worker does when it resumes
    # a finished detector pass. Existing classifications are replaced in
    # place rather than duplicated or sent through a classifier.
    reuse_detector_classes_in_json(path)

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["classification_categories"] == {"0": "fox"}
    assert result["images"][0]["detections"][0]["classifications"] == [["0", 0.8]]
    assert result["images"][1]["detections"] is None
    assert list(tmp_path.iterdir()) == [path]
