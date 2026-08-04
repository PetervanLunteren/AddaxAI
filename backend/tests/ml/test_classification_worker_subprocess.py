"""End-to-end run of the real classification worker as a subprocess.

Every other test in this area calls the worker's functions in-process or
replaces the classifier with a stub, so none of them touch the boundary
that actually ships: a JSON payload written by
`custom_classification_model.classify_detections`, handed to
`classification_worker.py` running as a separate process under a
different interpreter, which loads a model by dynamic import and writes
its answer back as JSON.

That boundary is entirely positional and untyped. A renamed payload key,
an argument passed in the wrong order to `_process_video_group`, or a
worker that never reads `scoring_detections` all produce a run that
completes and silently picks the wrong frame. In-process tests cannot
see any of it because they never serialise anything.

The model here is a fake `inference.py`: the worker only requires the
five methods `validate_interface` checks for, so no torch, no weights,
and no conda env are involved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")

WORKER = (
    Path(__file__).resolve().parents[2]
    / "app" / "ml" / "inference" / "classification_worker.py"
)

FAKE_MODEL = '''
from PIL import Image


class ModelInference:
    """Minimal model satisfying validate_interface. Returns a fixed
    distribution so the test can assert results came back at all."""

    def __init__(self, model_dir, model_path):
        self.model_dir = model_dir

    def check_gpu(self):
        return False

    def load_model(self):
        return None

    def get_crop(self, image, bbox):
        w, h = image.size
        x, y, bw, bh = bbox
        left, top = int(x * w), int(y * h)
        right, bottom = max(left + 1, int((x + bw) * w)), max(top + 1, int((y + bh) * h))
        return image.crop((left, top, right, bottom))

    def get_classification(self, crop):
        return [("1", 0.7), ("2", 0.3)]

    def get_class_names(self):
        return {"1": "badger", "2": "fox"}
'''


def _spawn_worker(tmp_path, payload):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "inference.py").write_text(FAKE_MODEL)
    weights = model_dir / "weights.pt"
    weights.write_bytes(b"not real weights")

    input_json = tmp_path / "in.json"
    output_json = tmp_path / "out.json"
    input_json.write_text(json.dumps(payload))

    proc = subprocess.run(
        [sys.executable, str(WORKER), str(model_dir), str(weights),
         str(input_json), str(output_json)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc, output_json


def _run_worker(tmp_path, payload):
    proc, output_json = _spawn_worker(tmp_path, payload)
    assert proc.returncode == 0, (
        f"worker exited {proc.returncode}\nSTDERR:\n{proc.stderr}"
    )
    return json.loads(output_json.read_text())


def test_worker_scores_the_best_frame_on_every_category(tmp_path, make_video):
    """The payload contract end to end. The only animal sits on frame 4,
    which is what the old animals-only scoring would have chosen. Two
    strong people on frame 16 outscore it, so a run that ignores
    `scoring_detections` picks 4 and this fails."""
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=24, fps=10)
    frames_dir = tmp_path / "frames" / "clip.mp4"

    payload = {
        # What gets classified: animals above the gate.
        "items": [
            {
                "source": "video",
                "video_path": str(video),
                "frame_number": 4,
                "bbox": [0.3, 0.3, 0.3, 0.3],
                "detection_conf": 0.55,
            }
        ],
        "best_frame_outputs": {str(video): str(frames_dir)},
        # What best-frame scoring runs on: every detection, any category.
        "scoring_detections": {
            str(video): [
                {"frame_number": 4, "conf": 0.55, "bbox": [0.3, 0.3, 0.3, 0.3]},
                {"frame_number": 16, "conf": 0.95, "bbox": [0.1, 0.1, 0.2, 0.2]},
                {"frame_number": 16, "conf": 0.90, "bbox": [0.6, 0.1, 0.2, 0.2]},
            ]
        },
    }

    out = _run_worker(tmp_path, payload)

    assert out["best_frames"] == {str(video): 16}
    assert (frames_dir / "frame000016.jpg").is_file()
    # This clip has a crop to classify, so it keeps the sequential walk.
    # Pinning the pixels here too keeps both routes honest.
    _assert_jpeg_shows_frame(frames_dir / "frame000016.jpg", 16)
    # Only the chosen frame is written, not one per candidate.
    assert sorted(p.name for p in frames_dir.glob("*.jpg")) == ["frame000016.jpg"]

    # The animal was still classified, on its own frame, in the same pass.
    assert len(out["results"]) == 1
    assert out["results"][0]["success"] is True
    assert out["class_names"] == {"1": "badger", "2": "fox"}


def _assert_jpeg_shows_frame(path, frame_number):
    """
    `make_video` encodes the frame index in the blue channel, so a written
    thumbnail can be checked against the frame it claims to be.

    Worth the four lines: asserting the *filename* alone passes happily
    if the frame is fetched from the wrong place, and fetching one frame
    without walking to it is exactly the change that could do that. Same
    tolerance as `test_video_iter.py`, loosened for lossy encoding.
    """
    from PIL import Image

    with Image.open(path) as img:
        blue = img.convert("RGB").getpixel((0, 0))[2]
    assert abs(blue - frame_number) <= 12, (
        f"{path.name} says frame {frame_number} but its pixels read {blue}"
    )


def test_worker_handles_a_video_with_nothing_to_classify(tmp_path, make_video):
    """A person-only clip: `items` is empty, so the worker classifies
    nothing, but must still produce a thumbnail anchored on the people.
    This is the exact shape that used to fall back to an arbitrary
    sharpness sample and leave the Labels grid with no usable card."""
    video = tmp_path / "people.mp4"
    make_video(video, total_frames=24, fps=10)
    frames_dir = tmp_path / "frames" / "people.mp4"

    payload = {
        "items": [],
        "best_frame_outputs": {str(video): str(frames_dir)},
        "scoring_detections": {
            str(video): [
                {"frame_number": 8, "conf": 0.60, "bbox": [0.1, 0.1, 0.2, 0.2]},
                {"frame_number": 20, "conf": 0.93, "bbox": [0.1, 0.1, 0.2, 0.2]},
            ]
        },
    }

    out = _run_worker(tmp_path, payload)

    assert out["best_frames"] == {str(video): 20}
    assert (frames_dir / "frame000020.jpg").is_file()
    _assert_jpeg_shows_frame(frames_dir / "frame000020.jpg", 20)
    assert out["results"] == []


def test_worker_falls_back_to_the_middle_frame_when_blank(tmp_path, make_video):
    """No detections anywhere. The worker still owes a thumbnail, and it
    is the middle frame rather than the first."""
    video = tmp_path / "blank.mp4"
    make_video(video, total_frames=20, fps=10)
    frames_dir = tmp_path / "frames" / "blank.mp4"

    out = _run_worker(
        tmp_path,
        {
            "items": [],
            "best_frame_outputs": {str(video): str(frames_dir)},
            "scoring_detections": {},
        },
    )

    assert out["best_frames"] == {str(video): 10}
    assert (frames_dir / "frame000010.jpg").is_file()
    _assert_jpeg_shows_frame(frames_dir / "frame000010.jpg", 10)


# ---------------------------------------------------------------------------
# Guards against a caller that disagrees with this worker
# ---------------------------------------------------------------------------


def test_worker_refuses_a_payload_with_videos_but_no_scoring_key(
    tmp_path, make_video
):
    """The failure this guard exists for, on 2026-07-31: a dev backend that
    had not restarted since an edit sent an old-shaped payload to a worker
    it had just spawned from new code on disk. Every video scored nothing,
    took the blank fallback, and got a middle-frame thumbnail with no
    detection on it. Downstream that reads as "the analysis found nothing":
    no cards in Labels, no observation, an "Empty" count card. Nothing
    errored, which is what made it expensive to diagnose.

    Failing the run is the right answer. The output is wrong either way,
    and only one of the two options says so.
    """
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=12, fps=10)

    proc, _ = _spawn_worker(
        tmp_path,
        {
            "items": [],
            "best_frame_outputs": {str(video): str(tmp_path / "frames")},
            # no "scoring_detections" key at all
        },
    )

    assert proc.returncode != 0
    assert "scoring_detections" in proc.stderr
    assert "older code" in proc.stderr


def test_worker_refuses_items_on_a_video_it_has_no_scoring_for(
    tmp_path, make_video
):
    """Crops queued for classification are themselves detections, so a video
    cannot have items and an empty scoring list. If it does, the two maps
    were built from different data or keyed differently. Both are keyed on a
    resolved absolute path assembled independently in the parent."""
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=12, fps=10)

    proc, _ = _spawn_worker(
        tmp_path,
        {
            "items": [
                {
                    "source": "video",
                    "video_path": str(video),
                    "frame_number": 4,
                    "bbox": [0.3, 0.3, 0.3, 0.3],
                    "detection_conf": 0.9,
                }
            ],
            "best_frame_outputs": {str(video): str(tmp_path / "frames")},
            # Key present, so the first guard passes, but this video is
            # missing from it (e.g. a path-keying mismatch).
            "scoring_detections": {"/some/other/path.mp4": []},
        },
    )

    assert proc.returncode != 0
    assert "no scoring detections" in proc.stderr


def test_worker_accepts_an_empty_scoring_map_when_there_are_no_videos(tmp_path):
    """Images only. No videos means nothing to score, so an empty map is
    correct and must not trip either guard."""
    out = _run_worker(tmp_path, {"items": [], "scoring_detections": {}})
    assert out["best_frames"] == {}
