"""Regression coverage for smoothing generic detector output safely."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "app" / "ml" / "smoothing_script.py"
)


def _install_smoother_stub(monkeypatch: pytest.MonkeyPatch, calls: list[tuple]):
    package = ModuleType("megadetector")
    package.__path__ = []  # type: ignore[attr-defined]
    postprocessing = ModuleType("megadetector.postprocessing")
    postprocessing.__path__ = []  # type: ignore[attr-defined]
    smoother = ModuleType(
        "megadetector.postprocessing.classification_postprocessing"
    )

    class Options:
        def __init__(self):
            self.propagate_classifications_through_taxonomy = False
            self.detection_confidence_threshold = None
            self.detection_category_names_to_smooth = []
            self.other_category_names = []
            self.modify_in_place = False

    def image_smooth(*, input_file, output_file, options):
        calls.append(("image", input_file, output_file, options))
        result = json.loads(json.dumps(input_file))
        result["image_smoothing_ran"] = True
        return result

    def sequence_smooth(
        *, input_file, cct_sequence_information, output_file, options
    ):
        calls.append(
            ("sequence", input_file, cct_sequence_information, output_file, options)
        )
        result = json.loads(json.dumps(input_file))
        result["sequence_smoothing_ran"] = True
        return result

    smoother.ClassificationSmoothingOptions = Options
    smoother.smooth_classification_results_image_level = image_smooth
    smoother.smooth_classification_results_sequence_level = sequence_smooth
    monkeypatch.setitem(sys.modules, "megadetector", package)
    monkeypatch.setitem(sys.modules, "megadetector.postprocessing", postprocessing)
    monkeypatch.setitem(
        sys.modules,
        "megadetector.postprocessing.classification_postprocessing",
        smoother,
    )


def _run_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    results: dict,
    options: dict,
) -> dict:
    input_path = tmp_path / "input.json"
    options_path = tmp_path / "options.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(results), encoding="utf-8")
    options_path.write_text(json.dumps(options), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT_PATH), str(input_path), str(options_path), str(output_path)],
    )
    namespace = runpy.run_path(str(_SCRIPT_PATH), run_name="smoothing_script_test")
    namespace["main"]()
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_custom_detector_categories_without_classifications_are_passed_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Detector class IDs are not classification labels for temporal smoothing."""
    calls: list[tuple] = []
    _install_smoother_stub(monkeypatch, calls)
    raw = {
        "images": [
            {
                "file": "image.jpg",
                "detections": [
                    {"category": "3", "conf": 0.95, "bbox": [0.1, 0.2, 0.3, 0.4]}
                ],
            }
        ],
        "detection_categories": {
            str(index): f"class_{index:02d}" for index in range(20)
        },
        "classification_categories": {},
    }

    output = _run_script(
        monkeypatch,
        tmp_path,
        raw,
        {"event_smoothing": True, "smoother_input": [{"seq_id": "one"}]},
    )

    assert output == raw
    assert calls == []


def test_classified_animal_results_keep_legacy_image_level_smoothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[tuple] = []
    _install_smoother_stub(monkeypatch, calls)
    raw = {
        "images": [
            {
                "file": "image.jpg",
                "detections": [
                    {
                        "category": "1",
                        "conf": 0.95,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                        "classifications": [["7", 0.8]],
                    }
                ],
            }
        ],
        "detection_categories": {"1": "animal"},
        "classification_categories": {"7": "deer"},
    }

    output = _run_script(monkeypatch, tmp_path, raw, {"event_smoothing": False})

    assert output["image_smoothing_ran"] is True
    assert len(calls) == 1
    assert calls[0][0] == "image"
    assert calls[0][3].detection_category_names_to_smooth == ["animal"]
