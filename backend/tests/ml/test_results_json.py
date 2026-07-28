"""Streaming readers for MegaDetector results JSON (app.ml.results_json).

These back the large-dataset DB load: iter_images must yield the same shapes
json.load did (notably plain floats, not Decimal), and read_top_level_object
must find a metadata key whether it sits before or after the images array.
"""

import json

from app.ml import results_json


def _write(tmp_path, payload, name="results.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_iter_images_yields_each_entry(tmp_path):
    payload = {
        "classification_categories": {"1": "deer"},
        "images": [
            {"file": "a.jpg", "detections": []},
            {"file": "b.jpg", "detections": []},
        ],
    }
    out = list(results_json.iter_images(_write(tmp_path, payload)))
    assert [img["file"] for img in out] == ["a.jpg", "b.jpg"]


def test_iter_images_numbers_are_float_not_decimal(tmp_path):
    """ijson defaults to Decimal; we force float to match json.load. A Decimal
    would break Float columns, arithmetic, and exif_metadata re-serialization."""
    payload = {
        "images": [
            {
                "file": "a.jpg",
                "detections": [
                    {"category": "1", "conf": 0.95, "bbox": [0.1, 0.2, 0.3, 0.4],
                     "classifications": [[1, 0.85]]},
                ],
            }
        ],
    }
    (img,) = list(results_json.iter_images(_write(tmp_path, payload)))
    det = img["detections"][0]
    assert type(det["conf"]) is float
    assert all(type(v) is float for v in det["bbox"])
    # Integer class id stays int; only the confidence is a float.
    assert det["classifications"][0][0] == 1
    assert type(det["classifications"][0][1]) is float


def test_read_top_level_object_before_and_after_images(tmp_path):
    cats = {"1": "deer", "2": "fox"}
    # Key after the images array (legacy merge writer order).
    after = _write(tmp_path, {"images": [{"file": "a"}],
                              "classification_categories": cats}, name="after.json")
    # Key before the images array (new merge writer order).
    before = _write(tmp_path, {"classification_categories": cats,
                               "images": [{"file": "a"}]}, name="before.json")
    assert results_json.read_top_level_object(after, "classification_categories") == cats
    assert results_json.read_top_level_object(before, "classification_categories") == cats


def test_read_top_level_object_absent_returns_empty(tmp_path):
    p = _write(tmp_path, {"images": [{"file": "a"}]})
    assert results_json.read_top_level_object(p, "classification_categories") == {}
