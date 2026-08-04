"""Best-frame selection scores every detection, whatever its category.

Until 2026-07-31 the classifier-fused path scored the best frame from the
population it was about to classify, which `extract_animal_detections`
restricts to category "1" above the classification gate. Two consequences:

1. A clip containing only people or only vehicles scored nothing, so the
   picker fell through to "sharpest of three evenly-spaced frames" and
   chose with no idea where the subject was. Combined with the grid only
   showing best-frame detections, such a video could end up showing no
   cards at all.
2. The same video got a different best frame depending on whether a
   classifier was configured, because the no-classifier path in
   `best_frame.py` always scored every category.

Detection confidence is the one signal every detector emits, so scoring on
it is a single rule that holds for any detector/classifier combination and
assumes no category vocabulary. A detector emitting `fish` / `shark` /
`turtle` needs no change.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from app.ml.json_pipeline import run_classification_on_json  # noqa: E402

# `classification_worker` imports its siblings by bare name, because in
# production it runs as a subprocess with its own directory on sys.path[0].
_INFERENCE_DIR = Path(__file__).resolve().parents[2] / "app" / "ml" / "inference"
sys.path.insert(0, str(_INFERENCE_DIR))

from app.ml import best_frame  # noqa: E402
from app.ml.best_frame import select_best_frames_streaming  # noqa: E402
from app.ml.inference.classification_worker import (  # noqa: E402
    _process_video_group,
)


def _det(category, conf, frame_number, bbox=(0.4, 0.4, 0.2, 0.2)):
    return {
        "category": category,
        "conf": conf,
        "bbox": list(bbox),
        "frame_number": frame_number,
    }


def _write_json(path, video_name, detections):
    path.write_text(
        json.dumps(
            {
                "detection_categories": {
                    "1": "animal",
                    "2": "person",
                    "3": "vehicle",
                },
                "images": [{"file": video_name, "detections": detections}],
            }
        )
    )


class _RecordingClassifier:
    """Captures both populations the pipeline hands the worker."""

    def __init__(self):
        self.items = None
        self.scoring = None

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
        self.items = items
        self.scoring = scoring_detections
        return [None] * len(items), {}, "cpu", {}


class _NeverClassifies:
    """Model double for the person-only case: no animal clears the gate,
    so nothing should ever be classified. Any call is a bug."""

    def get_crop(self, *a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("classification ran on a person-only video")

    def get_classification(self, *a, **kw):  # pragma: no cover
        raise AssertionError("classification ran on a person-only video")


def test_scoring_population_includes_every_category(tmp_path, make_video):
    """`items` stays animals-only (that is what gets classified), while
    `scoring_detections` carries person and vehicle too."""
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=12, fps=10)
    out = tmp_path / "detection_video.json"
    _write_json(
        out,
        "clip.mp4",
        [
            _det("1", 0.90, 0),   # animal, above the gate -> classified
            _det("2", 0.95, 4),   # person -> never classified
            _det("3", 0.80, 8),   # vehicle -> never classified
        ],
    )

    stub = _RecordingClassifier()
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

    assert [it["frame_number"] for it in stub.items] == [0]

    scored = stub.scoring[str(video.resolve())]
    assert sorted(d["frame_number"] for d in scored) == [0, 4, 8]
    assert sorted(d["conf"] for d in scored) == [0.80, 0.90, 0.95]


def test_person_only_video_picks_a_frame_the_person_is_on(tmp_path, make_video):
    """The regression that motivated this. No animal clears the gate, so
    `video_items` is empty and nothing is classified, but the person
    detections must still drive the pick instead of the blank-video
    sharpness fallback."""
    video = tmp_path / "people.mp4"
    make_video(video, total_frames=30, fps=10)
    dest = tmp_path / "out"

    # Two people on frame 20, one on frame 10. Frame 20 wins on summed
    # confidence. The blank fallback would have sampled 0, 10 and 20 and
    # picked on sharpness alone, which on a synthetic clip is arbitrary.
    scoring_dets = [
        {"frame_number": 10, "conf": 0.80, "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"frame_number": 20, "conf": 0.90, "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"frame_number": 20, "conf": 0.85, "bbox": [0.6, 0.1, 0.2, 0.2]},
    ]

    best = _process_video_group(
        _NeverClassifies(),
        str(video),
        [],                      # nothing to classify
        scoring_dets,
        dest,
        [],
        None,
        lambda _n: None,
    )

    assert best == 20
    assert (dest / "frame000020.jpg").is_file()


def test_undecodable_winner_falls_back_without_lying(tmp_path, make_video):
    """The frame is chosen from the JSON before anything is decoded, so
    the winner may turn out not to exist: containers over-report their
    frame count. What must never happen is a `best_frame_number` with a
    different frame's JPEG behind it, because the Labels grid would draw
    one moment's boxes over another moment's picture. The number and the
    file move together."""
    video = tmp_path / "short.mp4"
    make_video(video, total_frames=10, fps=10)
    dest = tmp_path / "out"

    scoring_dets = [
        {"frame_number": 4, "conf": 0.50, "bbox": [0.1, 0.1, 0.2, 0.2]},
        # Far past the last frame, and scoring higher, so it wins the
        # pick and then fails to decode.
        {"frame_number": 900, "conf": 0.99, "bbox": [0.1, 0.1, 0.2, 0.2]},
    ]

    best = _process_video_group(
        _NeverClassifies(),
        str(video),
        [],
        scoring_dets,
        dest,
        [],
        None,
        lambda _n: None,
    )

    assert best == 0
    assert (dest / "frame000000.jpg").is_file()
    assert not (dest / "frame000900.jpg").exists()


def test_blank_video_takes_the_middle_frame(tmp_path, make_video):
    """Nothing detected, so there is nothing to aim at. Position decides,
    and the middle beats the first frame because camera traps often open
    on the empty scene that triggered them."""
    video = tmp_path / "blank.mp4"
    make_video(video, total_frames=20, fps=10)
    dest = tmp_path / "out"

    best = _process_video_group(
        _NeverClassifies(),
        str(video),
        [],
        [],          # no detections at all
        dest,
        [],
        None,
        lambda _n: None,
    )

    assert best == 10
    assert (dest / "frame000010.jpg").is_file()


# ---------------------------------------------------------------------------
# The classifier-off path (best_frame.select_best_frames_streaming)
# ---------------------------------------------------------------------------


def test_both_paths_pick_the_same_frame(tmp_path, make_video):
    """`best_frame.py` runs when no classifier is configured and the
    worker runs when one is. They must agree, or the same footage gets a
    different thumbnail depending on a setting that has nothing to do
    with which frame looks best. They share `choose_frame_number`; this
    pins that they actually both call it.
    """
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=40, fps=10)

    detections = [
        _det("2", 0.40, 0),    # person, weak
        _det("1", 0.95, 20),   # animal, strongest
        _det("3", 0.60, 30),   # vehicle
    ]
    out = tmp_path / "detection_video.json"
    _write_json(out, "clip.mp4", detections)

    # Classifier-off path: opens the video itself, stamps the JSON.
    select_best_frames_streaming(out, tmp_path, tmp_path / "frames")
    from_json = json.loads(out.read_text())["images"][0]["best_frame_number"]

    # Classifier-on path: same detections as scoring input.
    scoring_dets = [
        {"frame_number": d["frame_number"], "conf": d["conf"], "bbox": d["bbox"]}
        for d in detections
    ]
    from_worker = _process_video_group(
        _NeverClassifies(),
        str(video),
        [],
        scoring_dets,
        tmp_path / "worker-frames",
        [],
        None,
        lambda _n: None,
    )

    assert from_json == from_worker == 20
    assert (tmp_path / "frames" / "clip.mp4" / "frame000020.jpg").is_file()
    assert (tmp_path / "worker-frames" / "frame000020.jpg").is_file()


def test_classifier_off_path_writes_only_the_chosen_frame(tmp_path, make_video):
    """The optimisation: decide from the JSON, then decode one frame. The
    old code decoded and held every frame carrying a detection so it
    could sharpness-score them, which on a 30-second clip meant tens of
    full-size images in memory at once."""
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=40, fps=10)
    out = tmp_path / "detection_video.json"
    _write_json(
        out,
        "clip.mp4",
        [_det("1", 0.50, 5), _det("1", 0.90, 15), _det("1", 0.60, 25)],
    )

    select_best_frames_streaming(out, tmp_path, tmp_path / "frames")

    written = sorted(p.name for p in (tmp_path / "frames" / "clip.mp4").glob("*.jpg"))
    assert written == ["frame000015.jpg"]


def test_result_is_identical_when_the_seek_is_refused(
    tmp_path, make_video, monkeypatch
):
    """
    Fetching one frame tries a seek first and walks the clip when the
    seek cannot be verified. That fallback is what makes the whole
    change safe on codecs nobody has tested, so it is pinned rather than
    assumed: force every seek to be refused and the answer must not
    move.
    """
    detections = [_det("1", 0.50, 5), _det("1", 0.90, 15), _det("1", 0.60, 25)]

    def run(folder):
        folder.mkdir()
        video = folder / "clip.mp4"
        make_video(video, total_frames=40, fps=10)
        out = folder / "detection_video.json"
        _write_json(out, "clip.mp4", detections)
        select_best_frames_streaming(out, folder, folder / "frames")
        frame_dir = folder / "frames" / "clip.mp4"
        written = sorted(p.name for p in frame_dir.glob("*.jpg"))
        return (
            json.loads(out.read_text())["images"][0]["best_frame_number"],
            written,
            (frame_dir / written[0]).read_bytes(),
        )

    with_seek = run(tmp_path / "seek")

    monkeypatch.setattr(best_frame, "read_frame_by_seek", lambda *a, **k: None)
    without_seek = run(tmp_path / "walk")

    assert with_seek[0] == without_seek[0] == 15
    assert with_seek[1] == without_seek[1] == ["frame000015.jpg"]
    assert with_seek[2] == without_seek[2], (
        "the seek and the walk produced different pixels for the same frame"
    )
