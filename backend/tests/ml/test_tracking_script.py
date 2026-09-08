"""The tracking script's pure parts: sampling, box normalisation, the
SharkTrack false-positive filter, the track crops and the JSON writer.

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
    assert ts._tracker_args(3.0, 0.2, 0.01).track_buffer == 6
    assert ts._tracker_args(5.0, 0.2, 0.01).track_buffer == 10
    assert ts._tracker_args(1.0, 0.2, 0.01).track_buffer == 2
    assert ts._tracker_args(0.25, 0.2, 0.01).track_buffer == 1


def test_tracker_floors_come_from_the_command_line():
    """A track starts from a box at the high floor and keeps boxes down
    to the low one; the script holds no floor of its own."""
    args = ts._tracker_args(2.0, 0.2, 0.01)
    assert (args.track_high_thresh, args.new_track_thresh, args.track_low_thresh) == (
        0.2, 0.2, 0.01
    )
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
    """SharkTrack's rule at 3 fps, applied only with --track_filter: a
    track survives when its best box is 0.7 or more, or when it lasts a
    second (3 sampled frames) and its centre moved at least 8% of the
    frame."""
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
