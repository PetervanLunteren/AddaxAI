"""
Tests for full-image video detection JSON synthesis.

`synthesize_full_image_video_json` fakes the MegaDetector video JSON so a
full-image classifier can process videos without running a detector. These
tests generate a tiny MP4 with the shared `make_video` fixture (no binary
fixture in the repo) and, for the end-to-end wiring, stub the classifier so
no real model or micromamba env is needed. Real frame decoding and
best-frame sharpness are covered by `test_video_iter.py` and the
classification worker's own tests.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from app.ml.full_image_detection import (  # noqa: E402
    synthesize_full_image_video_json,
)
from app.ml.json_pipeline import run_classification_on_json  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def test_video_synthesis_stamps_full_frame_detection_on_each_sampled_frame(
    tmp_path, make_video
):
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=20, fps=10)
    out = tmp_path / "detection_video.json"

    # Requested 5 fps against a 10 fps clip -> step 2 -> frames 0, 2, ..., 18.
    synthesize_full_image_video_json([video], tmp_path, out, fps=5)

    data = _load(out)
    assert data["detection_categories"] == {
        "1": "animal",
        "2": "person",
        "3": "vehicle",
    }
    assert len(data["images"]) == 1
    entry = data["images"][0]
    assert entry["file"] == "clip.mp4"
    assert entry["frame_rate"] == pytest.approx(10, abs=1)
    assert entry["frames_processed"] == list(range(0, 20, 2))

    dets = entry["detections"]
    assert [d["frame_number"] for d in dets] == list(range(0, 20, 2))
    assert all(
        d["category"] == "1"
        and d["conf"] == 1.0
        and d["bbox"] == [0.0, 0.0, 1.0, 1.0]
        for d in dets
    )


def test_video_synthesis_step_clamps_to_one_when_fps_exceeds_native(
    tmp_path, make_video
):
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=8, fps=10)
    out = tmp_path / "detection_video.json"

    # Requested fps far above native must not divide to a 0 step; every
    # frame is sampled instead.
    synthesize_full_image_video_json([video], tmp_path, out, fps=1000)

    entry = _load(out)["images"][0]
    assert entry["frames_processed"] == list(range(8))


def test_video_synthesis_records_failure_for_unopenable_video(tmp_path):
    out = tmp_path / "detection_video.json"
    broken = tmp_path / "nope.mp4"
    broken.write_bytes(b"not a video")

    synthesize_full_image_video_json([broken], tmp_path, out, fps=5)

    entry = _load(out)["images"][0]
    assert entry["file"] == "nope.mp4"
    assert entry["failure"] == "Failure video access"
    assert entry["detections"] is None
    assert entry["frame_rate"] == -1
    assert entry["frames_processed"] == []


class _StubResult:
    def __init__(self, probabilities: dict[str, float]):
        self.all_probabilities = probabilities


class _StubClassifier:
    """Records the items it receives and returns one deterministic result
    per item, matching the real `classify_detections` return contract
    (results, class_names, compute_device, best_frames)."""

    def __init__(self):
        self.seen_items: list[dict] | None = None
        self.seen_scoring: dict[str, list[dict]] | None = None
        self.device_callback = None

    def classify_detections(
        self,
        items,
        *,
        best_frame_outputs,
        scoring_detections,
        batch_size,
        progress_callback,
        device_callback=None,
        job_id,
    ):
        self.seen_items = items
        # Best-frame scoring runs on this, not on `items`: every detection
        # on the video, any category.
        self.seen_scoring = scoring_detections
        # Surface the device the moment the worker would report it, the
        # same as the real classifier. Exercises the report_device path.
        self.device_callback = device_callback
        if device_callback:
            device_callback("GPU (test)")
        results = [_StubResult({"cat": 0.8, "dog": 0.2}) for _ in items]
        class_names = {"1": "cat", "2": "dog"}
        # Best frame = first sampled frame of every known video.
        best_frames = {video_path: 0 for video_path in best_frame_outputs}
        return results, class_names, "cpu", best_frames


def test_synthesized_video_json_flows_through_classification(tmp_path, make_video):
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=12, fps=10)
    out = tmp_path / "detection_video.json"

    synthesize_full_image_video_json([video], tmp_path, out, fps=5)
    sampled = list(range(0, 12, 2))  # step 2

    stub = _StubClassifier()
    asyncio.run(
        run_classification_on_json(
            json_path=out,
            classification_model=stub,
            deployment_folder=tmp_path,
            batch_size=8,
            classification_gate=0.2,
            best_frame_output_base=tmp_path / "video_frames",
        )
    )

    # One classification item per sampled frame, all from the video.
    assert stub.seen_items is not None
    assert [it["frame_number"] for it in stub.seen_items] == sampled
    assert all(it["source"] == "video" for it in stub.seen_items)

    entry = _load(out)["images"][0]
    # Every synthesised detection got classifications written back.
    assert all("classifications" in d for d in entry["detections"])
    # Best frame number stamped from the classifier's map.
    assert entry["best_frame_number"] == 0
    # Classification surfaces its own device early (the fix for the modal
    # showing the previous phase's device, or "detecting..." for
    # full-image runs with no detector): the pipeline hands the
    # classifier a device_callback to fire the moment the device is known.
    assert callable(stub.device_callback)
