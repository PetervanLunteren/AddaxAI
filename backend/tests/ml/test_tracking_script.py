"""The tracking script's pure parts: sampling, box normalisation, the
SharkTrack false-positive filter and the JSON writer.

The detector and the tracker themselves need the marine env and real
frames; they are exercised by hand on footage (see the design's test
plan). Everything here runs on the dev interpreter.
"""

import json

import numpy as np

from app.ml.inference import tracking_script as ts


def test_sampling_matches_process_video():
    """0, stride, 2*stride ... with stride = round(native / fps), the
    frame indices process_video writes, so the app cannot tell the two
    runs apart."""
    assert ts.sampled_frames(frame_count=100, native_fps=30.0, fps=3.0) == list(range(0, 100, 10))
    assert ts.sampled_frames(frame_count=100, native_fps=29.97, fps=1.0) == list(range(0, 100, 30))
    # A rate above native, or an unknown native rate, samples every frame.
    assert ts.sampled_frames(frame_count=5, native_fps=3.0, fps=10.0) == [0, 1, 2, 3, 4]
    assert ts.sampled_frames(frame_count=3, native_fps=0.0, fps=3.0) == [0, 1, 2]
    assert ts.sampled_frames(frame_count=0, native_fps=30.0, fps=3.0) == []


def test_tracker_buffer_scales_with_the_sampling_rate():
    """Two seconds of lost track before it is dropped, at any fps."""
    assert ts._tracker_args(3.0).track_buffer == 6
    assert ts._tracker_args(5.0).track_buffer == 10
    assert ts._tracker_args(1.0).track_buffer == 2
    assert ts._tracker_args(0.25).track_buffer == 1
    args = ts._tracker_args(3.0)
    assert (args.track_high_thresh, args.track_low_thresh, args.new_track_thresh) == (0.4, 0.2, 0.4)
    assert args.match_thresh == 0.97 and args.with_reid is False


def test_track_rows_become_normalised_boxes_with_their_track_id():
    rows = np.array([
        # x1, y1, x2, y2, track_id, conf, cls, idx
        [100, 50, 300, 150, 7, 0.91, 0, 0],
        [-10, 0, 50, 2000, 8, 0.5, 1, 1],  # spills outside the frame
    ], dtype=np.float32)
    boxes = ts.normalise_track_rows(rows, width=1000, height=500, frame_number=90)
    assert boxes[0] == {
        "category": "0", "conf": 0.91, "bbox": [0.1, 0.1, 0.2, 0.2],
        "frame_number": 90, "track_id": 7,
    }
    # Clamped to the frame, never negative and never past 1.
    assert boxes[1]["bbox"] == [0.0, 0.0, 0.05, 1.0]
    assert boxes[1]["category"] == "1" and boxes[1]["track_id"] == 8
    assert ts.normalise_track_rows(np.zeros((0, 8)), 10, 10, 0) == []


def _track(track_id, frames, conf, x=0.1, drift=0.0):
    return [
        {"category": "0", "conf": conf, "bbox": [x + i * drift, 0.4, 0.1, 0.1],
         "frame_number": f, "track_id": track_id}
        for i, f in enumerate(frames)
    ]


def test_filter_keeps_confident_tracks_and_moving_long_ones():
    """SharkTrack's rule at 3 fps: a track survives when its best box is
    0.7 or more, or when it lasts a second (3 sampled frames) and its
    centre moved at least 8% of the frame."""
    confident_but_static = _track(1, [0, 10, 20], conf=0.75)
    long_and_moving = _track(2, [0, 10, 20, 30], conf=0.5, drift=0.05)
    short_and_moving = _track(3, [0, 10], conf=0.5, drift=0.2)
    long_but_static = _track(4, [0, 10, 20, 30], conf=0.5, drift=0.01)
    boxes = confident_but_static + long_and_moving + short_and_moving + long_but_static

    kept = {b["track_id"] for b in ts.filter_tracks(boxes, fps=3.0)}
    assert kept == {1, 2}


def test_filter_life_is_measured_in_sampled_frames_per_second():
    """One second is one frame at 1 fps and five at 5 fps."""
    two_frames = _track(1, [0, 30], conf=0.5, drift=0.2)
    assert {b["track_id"] for b in ts.filter_tracks(two_frames, fps=1.0)} == {1}
    assert ts.filter_tracks(two_frames, fps=5.0) == []


def test_filter_on_an_empty_video():
    assert ts.filter_tracks([], fps=3.0) == []


def test_results_json_has_the_shape_the_ingest_reads(tmp_path):
    out = tmp_path / "sub" / "video_results.json"
    ts.write_results(
        out,
        images=[
            {"file": "clip.mp4", "frame_rate": 30.0, "frames_processed": [0, 10],
             "detections": [{"category": "0", "conf": 0.9, "bbox": [0, 0, 1, 1],
                             "frame_number": 10, "track_id": 1}]},
            ts.failure_entry("broken.mp4"),
        ],
        categories={"0": "elasmobranch"},
        info={"detector": "sharktrack.pt", "tracker": "botsort"},
    )
    data = json.loads(out.read_text())
    assert data["detection_categories"] == {"0": "elasmobranch"}
    assert data["info"]["tracker"] == "botsort"
    assert "detection_completion_time" in data["info"]
    assert data["images"][0]["detections"][0]["track_id"] == 1
    failure = data["images"][1]
    assert failure["failure"] and failure["detections"] is None
    assert failure["frame_rate"] == -1 and failure["frames_processed"] == []
