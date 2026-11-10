"""The tracking script: sampling, box normalisation, the loop that hands
boxes to the tracker and keeps births, the SharkTrack false-positive
filter, the track crops and the JSON writer.

The detector needs a model env and real weights, so the loop test fakes
it; the tracker itself (``botsort.py``) runs for real, here and in
``test_botsort.py``. Everything runs on the dev interpreter.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from tqdm import tqdm

from app.ml.inference import tracking_script as ts


def test_sampling_takes_every_stride_th_frame_by_absolute_index():
    """0, stride, 2*stride ... with stride = round(native / fps), numbered
    by source frame index as process_video numbered them."""
    assert ts.sampled_frames(frame_count=100, native_fps=30.0, fps=3.0) == list(range(0, 100, 10))
    assert ts.sampled_frames(frame_count=100, native_fps=29.97, fps=1.0) == list(range(0, 100, 30))
    # A rate above native, or an unknown native rate, samples every frame.
    assert ts.sampled_frames(frame_count=5, native_fps=3.0, fps=10.0) == [0, 1, 2, 3, 4]
    assert ts.sampled_frames(frame_count=3, native_fps=0.0, fps=3.0) == [0, 1, 2]
    assert ts.sampled_frames(frame_count=0, native_fps=30.0, fps=3.0) == []


def test_the_catalogs_class_mapping_outranks_the_packages_own_names():
    """SharkTrack is a plain ultralytics checkpoint, so the megadetector
    package cannot name its class: left alone it would assert MegaDetector's
    three, add one, and store every shark as "animal". The catalog's mapping
    is what the run reports, and it wins over the package's default so the
    names never depend on the global we set beside it."""
    md_default = {"1": "animal", "2": "person", "3": "vehicle"}

    # SharkTrack: the catalog decides, ids and names both.
    assert ts.pick_categories({"0": "elasmobranch"}, None, md_default) == {
        "0": "elasmobranch"
    }
    # RF-DETR names itself, so it is used when the catalog says nothing.
    assert ts.pick_categories(None, {"0": "fish"}, md_default) == {"0": "fish"}
    # A MegaDetector .pt has neither, and falls back to the package's three.
    assert ts.pick_categories(None, None, md_default) == md_default
    # Whatever the source, the JSON carries strings on both sides.
    assert ts.pick_categories(None, {0: "fish"}, md_default) == {"0": "fish"}


def test_tracker_buffer_scales_with_the_sampling_rate():
    """Two seconds of lost track before it is dropped, at any fps."""
    assert ts.track_buffer(3.0) == 6
    assert ts.track_buffer(5.0) == 10
    assert ts.track_buffer(1.0) == 2
    assert ts.track_buffer(0.25) == 1


def test_tracker_floors_come_from_the_command_line():
    """A track starts from a box at the high floor and keeps boxes down
    to the low one; the script holds no floor of its own."""
    tracker = ts.make_tracker(2.0, 0.2, 0.01, 3840)
    assert (tracker.high_thresh, tracker.low_thresh) == (0.2, 0.01)
    assert tracker.match_thresh == 0.97
    assert tracker.max_time_lost == 4
    assert tracker.gmc.downscale == 4


def test_track_rows_become_normalised_boxes_with_their_track_id():
    rows = np.array([
        # x1, y1, x2, y2, track_id, conf, cls
        [100, 50, 300, 150, 7, 0.91, 0],
        [-10, 0, 50, 2000, 8, 0.5, 1],  # spills outside the frame
    ], dtype=np.float32)
    boxes = ts.normalise_track_rows(rows, width=1000, height=500, frame_number=90)
    assert boxes[0] == {
        "category": "0", "conf": 0.91, "bbox": [0.1, 0.1, 0.2, 0.2],
        "frame_number": 90, "track_id": 7,
    }
    # Clamped to the frame, never negative and never past 1.
    assert boxes[1]["bbox"] == [0.0, 0.0, 0.05, 1.0]
    assert boxes[1]["category"] == "1" and boxes[1]["track_id"] == 8
    assert ts.normalise_track_rows(np.zeros((0, 7)), 10, 10, 0) == []


def test_raw_detection_rows_keep_every_box_and_carry_no_track():
    """--raw_detections_json writes what the detector saw before the
    tracker chose: the same box shape, no track_id, nothing dropped."""
    rows = np.array([
        # x1, y1, x2, y2, conf, cls
        [100, 50, 300, 150, 0.91, 0],
        [0, 0, 10, 10, 0.01, 2],  # the storage floor, kept
    ], dtype=np.float32)
    boxes = ts.normalise_detection_rows(rows, width=1000, height=500, frame_number=90)
    assert boxes == [
        {"category": "0", "conf": 0.91, "bbox": [0.1, 0.1, 0.2, 0.2], "frame_number": 90},
        {"category": "2", "conf": 0.01, "bbox": [0.0, 0.0, 0.01, 0.02], "frame_number": 90},
    ]
    assert ts.normalise_detection_rows(np.zeros((0, 6)), 10, 10, 0) == []


def _track(track_id, frames, conf, x=0.1, drift=0.0):
    return [
        {"category": "0", "conf": conf, "bbox": [x + i * drift, 0.4, 0.1, 0.1],
         "frame_number": f, "track_id": track_id}
        for i, f in enumerate(frames)
    ]


UNDERWATER = dict(min_motion=0.08, exempt_conf=0.7)  # SharkTrack's values
CAMERA_TRAP = dict(min_motion=0.06, exempt_conf=0.5)


def test_filter_keeps_confident_tracks_and_moving_long_ones():
    """SharkTrack's rule at 3 fps with its own values: a track survives
    when its best box is 0.7 or more, or when it lasts a second (3
    sampled frames) and its centre moved at least 8% of the frame."""
    confident_but_static = _track(1, [0, 10, 20], conf=0.75)
    long_and_moving = _track(2, [0, 10, 20, 30], conf=0.5, drift=0.05)
    short_and_moving = _track(3, [0, 10], conf=0.5, drift=0.2)
    long_but_static = _track(4, [0, 10, 20, 30], conf=0.5, drift=0.01)
    boxes = confident_but_static + long_and_moving + short_and_moving + long_but_static

    kept = {b["track_id"] for b in ts.filter_tracks(boxes, fps=3.0, **UNDERWATER)}
    assert kept == {1, 2}


def test_the_camera_trap_values_keep_a_still_animal_the_underwater_ones_drop():
    """A resting animal at 0.55 for the whole clip barely moves. Under
    water that is a false box and goes; on a camera trap it is ordinary
    and the lower exemption keeps it. The benchmark measured the
    difference as one resting-animal clip in ten emptied."""
    resting = _track(1, [0, 10, 20, 30, 40], conf=0.55, drift=0.005)
    assert ts.filter_tracks(resting, fps=3.0, **UNDERWATER) == []
    assert {b["track_id"] for b in ts.filter_tracks(resting, fps=3.0, **CAMERA_TRAP)} == {1}
    # Below the camera-trap exemption a static track still needs 6% of motion.
    weak_and_static = _track(2, [0, 10, 20, 30, 40], conf=0.4, drift=0.005)
    assert ts.filter_tracks(weak_and_static, fps=3.0, **CAMERA_TRAP) == []
    weak_but_walking = _track(3, [0, 10, 20, 30, 40], conf=0.4, drift=0.02)
    kept = ts.filter_tracks(weak_but_walking, fps=3.0, **CAMERA_TRAP)
    assert {b["track_id"] for b in kept} == {3}


def test_filter_life_is_measured_in_sampled_frames_per_second():
    """One second is one frame at 1 fps and five at 5 fps."""
    two_frames = _track(1, [0, 30], conf=0.5, drift=0.2)
    assert {b["track_id"] for b in ts.filter_tracks(two_frames, fps=1.0, **UNDERWATER)} == {1}
    assert ts.filter_tracks(two_frames, fps=5.0, **UNDERWATER) == []


def test_filter_on_an_empty_video():
    assert ts.filter_tracks([], fps=3.0, **UNDERWATER) == []


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


def test_motion_compensation_works_on_about_a_thousand_pixels():
    """4K shrinks by 4, 1080p by 2 (the tracker's own default), and a
    small clip is never shrunk below the default either."""
    assert ts.gmc_downscale(3840) == 4
    assert ts.gmc_downscale(1920) == 2
    assert ts.gmc_downscale(1280) == 2
    assert ts.gmc_downscale(640) == 2
    assert ts.gmc_downscale(0) == 2


def test_decode_size_caps_the_long_edge_and_keeps_even_sides():
    """1920 on the long edge, the cover frame's own cap, so a track card
    cut from a decoded frame matches a card cut from the cover. A
    portrait clip is capped by its height."""
    assert ts.decode_size(3840, 2160) == (1920, 1080)
    assert ts.decode_size(2160, 3840) == (1080, 1920)
    assert ts.decode_size(1920, 1080) == (1920, 1080)
    # Smaller than the cap: left alone, apart from making the sides even.
    assert ts.decode_size(640, 360) == (640, 360)
    assert ts.decode_size(1279, 721) == (1278, 720)
    assert ts.decode_size(0, 0) == (0, 0)


def _frame(width=200, height=100, colour=(10, 20, 30)):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = colour  # BGR
    return frame


def _decode(jpeg: bytes):
    import io

    from PIL import Image

    return Image.open(io.BytesIO(jpeg))


def test_track_crop_is_a_padded_square_capped_at_the_card_size():
    """The crop service's geometry: a square around the box with 10%
    context, shrunk to at most 512 px, never enlarged."""
    from PIL import Image

    frame = Image.new("RGB", (2000, 1000), (0, 0, 0))
    # A 100x50 box: the square is 100 + 2*10 = 120 px a side.
    small = _decode(ts.cut_track_crop(frame, [0.1, 0.1, 0.05, 0.05]))
    assert small.size == (120, 120)
    # A box taller than 512 px lands at 512.
    big = _decode(ts.cut_track_crop(frame, [0.1, 0.0, 0.5, 1.0]))
    assert max(big.size) == 512


def test_best_crop_follows_the_highest_confidence_and_keeps_the_earliest_on_a_tie():
    """The crop must sit on the frame `summarise_tracks` names as the
    representative when it reads the JSON back: the highest rounded
    confidence, ties to the earliest frame. So a later box replaces the
    crop only when strictly better."""
    best: dict = {}
    box = {"track_id": 1, "conf": 0.5, "bbox": [0.1, 0.1, 0.2, 0.2]}
    ts.keep_best_crop(best, [box], _frame(colour=(0, 0, 255)), 10)
    ts.keep_best_crop(best, [dict(box, conf=0.5)], _frame(colour=(0, 255, 0)), 20)
    assert best[1][:2] == (0.5, 10)
    ts.keep_best_crop(best, [dict(box, conf=0.9)], _frame(colour=(255, 0, 0)), 30)
    assert best[1][:2] == (0.9, 30)
    # The crop is that frame's pixels (BGR blue becomes RGB blue), within
    # JPEG rounding.
    r, g, b = _decode(best[1][2]).convert("RGB").getpixel((5, 5))
    assert (r, g) == (0, 0) and b >= 250


def test_only_surviving_tracks_get_their_crop_written(tmp_path):
    best = {
        1: (0.9, 10, b"one"),
        2: (0.4, 20, b"two"),
    }
    ts.write_track_crops(tmp_path / "clip.mp4", best, surviving={1})
    assert (tmp_path / "clip.mp4" / "track000001.jpg").read_bytes() == b"one"
    assert not (tmp_path / "clip.mp4" / "track000002.jpg").exists()


def test_ffmpeg_command_selects_by_source_index_and_scales():
    """Output frame i must be source frame i * stride, as process_video
    numbers them: select by frame index, never by time, and passthrough
    so ffmpeg never pads the timeline with duplicates."""
    from pathlib import Path

    cmd = ts.ffmpeg_decode_cmd("/env/bin/ffmpeg", Path("/v/clip.mp4"), 10, 1920, 1080)
    assert cmd[0] == "/env/bin/ffmpeg"
    assert cmd[cmd.index("-i") + 1] == "/v/clip.mp4"
    assert cmd[cmd.index("-vf") + 1] == "select=not(mod(n\\,10)),scale=1920:1080"
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert cmd[cmd.index("-hwaccel") + 1] == "auto"
    assert cmd[-4:] == ["-f", "rawvideo", "-pix_fmt", "bgr24"][:0] + cmd[-4:]
    assert cmd[cmd.index("-pix_fmt") + 1] == "bgr24" and cmd[-1] == "-"


def test_sampling_stride():
    assert ts.sampling_stride(29.97, 3.0) == 10
    assert ts.sampling_stride(30.0, 1.0) == 30
    assert ts.sampling_stride(0.0, 3.0) == 1


def test_ffmpeg_frames_number_the_frames_like_process_video(tmp_path, make_video):
    """A real pipe on a 20-frame clip whose blue channel is the frame
    index: with a stride of 5 the pipe yields frames 0, 5, 10, 15 and
    each carries its own index, so the numbering survives the decode."""
    import shutil

    import pytest

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    video = tmp_path / "tiny.mp4"
    make_video(video, total_frames=20)

    frames = list(ts.ffmpeg_frames(ffmpeg, video, 5, 64, 48))
    assert [n for n, _ in frames] == [0, 5, 10, 15]
    for n, frame in frames:
        assert frame.shape == (48, 64, 3)
        # Blue channel is the source frame index (within codec noise).
        assert abs(int(frame[24, 32, 0]) - n) <= 4, (n, int(frame[24, 32, 0]))


class _ScriptedDetector:
    """Hands the tracker the rows scripted for each sampled frame, in the
    detector's own format: float32 x1, y1, x2, y2, conf, cls in pixels."""

    def __init__(self, script: dict[int, list[list[float]]]):
        self.script = script
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        rows = self.script.get(self.calls, [])
        self.calls += 1
        return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def _run_loop(tmp_path: Path, make_video, script: dict[int, list[list[float]]]):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    video = tmp_path / "clip.mp4"
    make_video(video, total_frames=12, fps=10, size=(64, 48))
    # 10 fps sampled at 10 fps: every frame, so call n is frame n.
    return ts.track_video(
        _ScriptedDetector(script), 0.5, 0.01, ffmpeg, video, 10.0, 10.0, (64, 48),
        tqdm(disable=True), filter_motion=0.0, filter_exempt=0.0,
    )


def _walker(first: int, last: int, conf: float = 0.9) -> dict[int, list[list[float]]]:
    """One box drifting a pixel a frame from frame ``first`` to ``last``."""
    return {f: [[10 + f, 10, 30 + f, 30, conf, 0]] for f in range(first, last + 1)}


def test_the_first_frame_of_a_video_confirms_a_track_at_once(tmp_path, make_video):
    processed, boxes, crops = _run_loop(tmp_path, make_video, _walker(0, 11))
    assert processed == list(range(12))
    assert [b["frame_number"] for b in boxes] == list(range(12))
    assert {b["track_id"] for b in boxes} == {1}
    assert set(crops) == {1}


def test_a_track_born_later_stores_its_first_box_once_confirmed(tmp_path, make_video):
    """A track that starts on frame 3 is reported from frame 4; the box it
    started from is stored then, in frame order, and only once."""
    _, boxes, _ = _run_loop(tmp_path, make_video, _walker(3, 11))
    frames = [b["frame_number"] for b in boxes]
    assert frames == list(range(3, 12))
    # The stored box is the detector's own, not a filtered estimate.
    assert boxes[0]["bbox"] == pytest.approx([13 / 64, 10 / 48, 20 / 64, 20 / 48], abs=1e-4)


def test_a_box_never_confirmed_is_never_stored(tmp_path, make_video):
    """A one-frame box starts a track that is dropped on the next frame,
    so nothing of it reaches the JSON. Its crop may be cut while its frame
    is in memory; the caller writes crops for surviving tracks only."""
    script = _walker(0, 11)
    script[6] = [*script[6], [40, 30, 60, 46, 0.9, 0]]
    _, boxes, _ = _run_loop(tmp_path, make_video, script)
    assert {b["track_id"] for b in boxes} == {1}
    assert len(boxes) == 12


def test_results_are_written_verified_and_never_half(tmp_path, monkeypatch):
    """A write the OS reports as done can land truncated on a failing
    drive. The script parses its own file back before it renames it over
    the real name, retries, and gives up with an error rather than leave
    a results file the ingest cannot read."""
    out = tmp_path / "video_results.json"
    real_dump = json.dump
    calls = {"n": 0}

    def truncating_dump(obj, f, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            f.write('{"images": [')  # a half file, as a failing drive leaves it
            return
        real_dump(obj, f, **kw)

    monkeypatch.setattr(ts.json, "dump", truncating_dump)
    ts.write_results(out, images=[], categories={"0": "animal"}, info={})

    assert calls["n"] == 2
    assert json.loads(out.read_text())["detection_categories"] == {"0": "animal"}
    assert not (tmp_path / "video_results.json.partial").exists()

    monkeypatch.setattr(ts.json, "dump", lambda obj, f, **kw: f.write("{"))
    with pytest.raises(OSError, match="readable results file"):
        ts.write_results(tmp_path / "never.json", images=[], categories={}, info={})
    assert not (tmp_path / "never.json").exists()
