"""
Tests for the streaming video frame iterator.

Generates a tiny MP4 on the fly with cv2 (avoids carrying a binary
fixture in the repo) and pins the contract `iter_wanted_frames` makes
to the classification worker.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.ml.inference.video_iter import (  # noqa: E402
    iter_wanted_frames,
    open_video,
    sample_indices,
)


def _make_video(path, total_frames: int, fps: int = 10, size: tuple[int, int] = (64, 48)) -> None:
    """
    Render a deterministic test video. Each frame is a solid colour
    whose blue channel encodes the frame index, so the test can verify
    `iter_wanted_frames` yielded the right indices by looking at the
    pixel values.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not open mp4v encoder on this machine")
    try:
        for i in range(total_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            frame[:, :, 0] = i  # OpenCV writes BGR, so channel 0 = blue
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def tiny_video(tmp_path):
    video_path = tmp_path / "tiny.mp4"
    _make_video(video_path, total_frames=20)
    return video_path


def test_open_video_returns_capture_positioned_at_frame_zero(tiny_video):
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        success, frame = cap.read()
        assert success
        # Frame 0 was rendered with blue=0.
        assert int(frame[0, 0, 0]) == 0
    finally:
        cap.release()


def test_open_video_returns_none_for_missing_file(tmp_path):
    assert open_video(tmp_path / "does-not-exist.mp4") is None


def test_iter_wanted_frames_yields_requested_indices_in_order(tiny_video):
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        wanted = {0, 5, 12, 19}
        yielded = list(iter_wanted_frames(cap, wanted))
    finally:
        cap.release()

    assert [fn for fn, _ in yielded] == [0, 5, 12, 19]
    # Pixel encoding round-trip: blue channel of each yielded frame
    # must equal that frame's index. mp4v compression is lossy so allow
    # a generous tolerance — we're verifying we got roughly the right
    # frame, not the exact value.
    for fn, image in yielded:
        rgb = np.array(image)
        # PIL.Image is RGB; the blue channel is index 2.
        assert abs(int(rgb[0, 0, 2]) - fn) <= 12


def test_iter_wanted_frames_empty_set_yields_nothing(tiny_video):
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        yielded = list(iter_wanted_frames(cap, set()))
    finally:
        cap.release()
    assert yielded == []


def test_iter_wanted_frames_stops_at_end_of_stream_even_if_index_unreachable(tiny_video):
    """A wanted index past the actual decodable end is silently dropped."""
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        wanted = {0, 9999}
        yielded = list(iter_wanted_frames(cap, wanted))
    finally:
        cap.release()
    assert [fn for fn, _ in yielded] == [0]


def test_iter_wanted_frames_stops_early_when_last_wanted_reached(tiny_video):
    """Should not read past the highest requested index."""
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        yielded = list(iter_wanted_frames(cap, {2}))
    finally:
        cap.release()
    assert [fn for fn, _ in yielded] == [2]


def test_sample_indices_returns_evenly_spaced_indices():
    assert sample_indices(total=10, count=3) == [0, 3, 6]
    assert sample_indices(total=2, count=3) == [0, 1]
    assert sample_indices(total=0, count=3) == []
    assert sample_indices(total=10, count=0) == []
