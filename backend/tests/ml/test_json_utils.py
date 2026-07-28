"""Tests for app.ml.json_utils, with a focus on tolerating failed-video
entries written by MegaDetector's process_video when a video can't be
decoded (corrupt file, unsupported codec, no retrievable frames).

A failed-video entry looks like:

    {
        "file": "video.mp4",
        "frame_rate": -1.0,
        "frames_processed": [],
        "failure": "Failure processing video: Error: found no frames in ...",
        "detections": null
    }

Naive `.get("detections", [])` returns None (not []) when the key is
present with value None, so iteration crashes with TypeError. These
tests pin the defensive iteration that fixes that.
"""

from app.ml.json_utils import (
    assign_uuids_to_detection_json,
    collect_md_failures,
    extract_animal_detections,
    trim_classification_results,
)


def _ok_image(file: str = "ok.jpg", category: str = "1") -> dict:
    return {
        "file": file,
        "detections": [
            {"category": category, "conf": 0.9, "bbox": [0.0, 0.0, 0.1, 0.1]},
        ],
    }


def _failed_video(file: str = "corrupt.mp4") -> dict:
    """Shape of an entry emitted by megadetector.detection.process_video
    when it cannot decode a video. `detections` is explicitly null."""
    return {
        "file": file,
        "frame_rate": -1.0,
        "frames_processed": [],
        "failure": "Failure processing video: Error: found no frames in file",
        "detections": None,
    }


def test_extract_animal_detections_skips_failure_entries() -> None:
    """Pre-fix this raised TypeError because `.get('detections', [])`
    returned None and the inner loop tried to iterate None."""
    md = {
        "images": [
            _ok_image("a.jpg"),
            _failed_video("corrupt.mp4"),
            _ok_image("b.jpg"),
        ],
    }
    animals = extract_animal_detections(md, min_confidence=0.1)
    # Both ok images contribute one animal detection each. The corrupt
    # entry contributes nothing and does not raise.
    assert len(animals) == 2
    img_indices = {idx for idx, _, _ in animals}
    assert img_indices == {0, 2}


def test_extract_animal_detections_handles_top_level_null_images() -> None:
    """`{"images": null}` should be treated as an empty list, not crash."""
    assert extract_animal_detections({"images": None}, min_confidence=0.1) == []
    assert extract_animal_detections({}, min_confidence=0.1) == []


def test_extract_animal_detections_ignores_non_animal_categories() -> None:
    """Sanity: only category '1' is returned."""
    md = {
        "images": [
            {
                "file": "a.jpg",
                "detections": [
                    {"category": "1", "conf": 0.9, "bbox": [0, 0, 0.1, 0.1]},
                    {"category": "2", "conf": 0.9, "bbox": [0, 0, 0.1, 0.1]},
                    {"category": "3", "conf": 0.9, "bbox": [0, 0, 0.1, 0.1]},
                ],
            },
        ],
    }
    animals = extract_animal_detections(md, min_confidence=0.1)
    assert len(animals) == 1


def test_extract_animal_detections_applies_classification_gate() -> None:
    """Animal detections below the gate are not sent to the classifier.
    MD runs untresholded (0.005), so the JSON carries a near-noise tail
    that must be gated here."""
    md = {
        "images": [
            {
                "file": "a.jpg",
                "detections": [
                    {"category": "1", "conf": 0.9, "bbox": [0, 0, 0.1, 0.1]},
                    {"category": "1", "conf": 0.1, "bbox": [0, 0, 0.1, 0.1]},
                    {"category": "1", "conf": 0.02, "bbox": [0, 0, 0.1, 0.1]},
                ],
            },
        ],
    }
    # Exactly-at-gate passes; below-gate is skipped.
    animals = extract_animal_detections(md, min_confidence=0.1)
    assert [d["conf"] for _, _, d in animals] == [0.9, 0.1]
    # Lowering the gate brings the tail into classification scope.
    animals_low = extract_animal_detections(md, min_confidence=0.005)
    assert len(animals_low) == 3


def test_collect_md_failures_returns_failure_entries() -> None:
    md = {
        "images": [
            _ok_image("a.jpg"),
            _failed_video("corrupt.mp4"),
            _failed_video("bad.mp4"),
        ],
    }
    failures = collect_md_failures(md)
    assert len(failures) == 2
    assert failures[0]["file"] == "corrupt.mp4"
    assert "Failure" in failures[0]["reason"]
    assert failures[1]["file"] == "bad.mp4"


def test_collect_md_failures_empty_when_all_ok() -> None:
    md = {"images": [_ok_image("a.jpg"), _ok_image("b.jpg")]}
    assert collect_md_failures(md) == []


def test_collect_md_failures_handles_top_level_null_images() -> None:
    assert collect_md_failures({"images": None}) == []
    assert collect_md_failures({}) == []


def test_trim_classification_results_skips_failure_entries() -> None:
    """trim_classification_results is called in the worker between phase 5
    (JSON merge) and phase 6 (DB load). It iterates every image's
    detections, so a failure entry with `detections: null` would crash it
    pre-fix. Mirrors the exact failure mode in the production bug."""
    md = {
        "classification_categories": {"1": "leopard", "2": "lion"},
        "images": [
            {
                "file": "a.mp4",
                "detections": [
                    {
                        "category": "1",
                        "conf": 0.9,
                        "bbox": [0, 0, 0.1, 0.1],
                        "classifications": [["1", 0.8]],
                    },
                ],
            },
            _failed_video("corrupt.mp4"),
        ],
    }
    # Pre-fix this raised TypeError. Post-fix it returns without raising,
    # and the failure entry is left untouched.
    removed = trim_classification_results(md, max_classifications=5)
    assert removed == 1  # "lion" (id "2") was unreferenced, got pruned
    assert md["images"][1]["detections"] is None


def test_assign_uuids_skips_failure_entries_safely() -> None:
    """Failed-video entries have `detections: null`; assigning detection
    IDs must not crash on them."""
    md = {
        "images": [
            _ok_image("a.jpg"),
            _failed_video("corrupt.mp4"),
        ],
    }
    assign_uuids_to_detection_json(md)
    # ok image got file_id + detection_id assigned
    assert "file_id" in md["images"][0]
    assert "detection_id" in md["images"][0]["detections"][0]
    # failed entry got file_id assigned, detections is still None
    # (we don't touch it; the user-facing warning surfaces it elsewhere)
    assert "file_id" in md["images"][1]
    assert md["images"][1]["detections"] is None
