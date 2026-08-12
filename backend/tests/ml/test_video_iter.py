"""
Tests for the streaming video frame iterator.

Generates a tiny MP4 on the fly with cv2 (avoids carrying a binary
fixture in the repo) and pins the contract `iter_wanted_frames` makes
to the classification worker.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from app.ml.inference.video_iter import (  # noqa: E402
    iter_wanted_frames,
    open_video,
    read_frame_by_seek,
    sample_indices,
)


@pytest.fixture
def tiny_video(tmp_path, make_video):
    video_path = tmp_path / "tiny.mp4"
    make_video(video_path, total_frames=20)
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
        yielded = list(iter_wanted_frames(cap, wanted, tiny_video))
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
        yielded = list(iter_wanted_frames(cap, set(), tiny_video))
    finally:
        cap.release()
    assert yielded == []


def test_iter_wanted_frames_stops_at_end_of_stream_even_if_index_unreachable(tiny_video):
    """A wanted index past the actual decodable end is silently dropped."""
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        wanted = {0, 9999}
        yielded = list(iter_wanted_frames(cap, wanted, tiny_video))
    finally:
        cap.release()
    assert [fn for fn, _ in yielded] == [0]


def test_iter_wanted_frames_stops_early_when_last_wanted_reached(tiny_video):
    """Should not read past the highest requested index."""
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        yielded = list(iter_wanted_frames(cap, {2}, tiny_video))
    finally:
        cap.release()
    assert [fn for fn, _ in yielded] == [2]


def test_sample_indices_returns_evenly_spaced_indices():
    assert sample_indices(total=10, count=3) == [0, 3, 6]
    assert sample_indices(total=2, count=3) == [0, 1]
    assert sample_indices(total=0, count=3) == []
    assert sample_indices(total=10, count=0) == []


# ---------------------------------------------------------------------------
# read_frame_by_seek
#
# The contract that matters is "the frame you get back is the frame you
# asked for, or you get nothing". Everything here is written against the
# sequential walk, which is the definition of the right answer, so a seek
# that silently lands elsewhere fails rather than passing on a filename.
# ---------------------------------------------------------------------------

# ("mp4v", ".mp4") is what every other video test uses. MJPG/.avi is here
# because it is the shape DEVELOPERS.md calls out as the awkward one (the
# Browning cameras that write MJPG AVI with no capture date).
CODECS = [("mp4v", ".mp4"), ("MJPG", ".avi")]


def _walk_to(cap, frame_number, video_path):
    """Ground truth: the frame the sequential path would have produced."""
    for num, image in iter_wanted_frames(cap, {frame_number}, video_path):
        if num == frame_number:
            return image
    return None


@pytest.mark.parametrize("codec,suffix", CODECS)
def test_read_frame_by_seek_returns_the_frame_the_walk_would_have(
    tmp_path, make_video, codec, suffix
):
    """Pixel equality against the walk, at several points in the clip."""
    video = tmp_path / f"clip{suffix}"
    make_video(video, total_frames=30, codec=codec)

    for target in (0, 1, 7, 15, 29):
        seek_cap = open_video(video)
        walk_cap = open_video(video)
        assert seek_cap is not None and walk_cap is not None
        try:
            seeked = read_frame_by_seek(seek_cap, target, 30)
            walked = _walk_to(walk_cap, target, video)
        finally:
            seek_cap.release()
            walk_cap.release()

        assert walked is not None, f"walk could not reach frame {target}"
        # A refused seek is allowed (the caller falls back), a wrong one
        # is not.
        if seeked is not None:
            assert list(seeked.getdata()) == list(walked.getdata()), (
                f"{codec}: seek to frame {target} returned different pixels "
                f"than walking to it"
            )


@pytest.mark.parametrize("codec,suffix", CODECS)
def test_read_frame_by_seek_fires_on_the_common_case(
    tmp_path, make_video, codec, suffix
):
    """
    The whole point is the middle frame of a blank clip. If this ever
    starts returning None the fix still works but has stopped paying,
    which is worth failing over rather than silently regressing.
    """
    video = tmp_path / f"clip{suffix}"
    make_video(video, total_frames=30, codec=codec)
    cap = open_video(video)
    assert cap is not None
    try:
        assert read_frame_by_seek(cap, 15, 30) is not None
    finally:
        cap.release()


def test_read_frame_by_seek_refuses_out_of_range(tiny_video):
    """
    The range guard, which is the real safety net. Frame 900 of a
    20-frame clip must be refused from the frame count alone, without
    depending on how a backend clamps an out-of-range seek. This is the
    case `test_undecodable_winner_falls_back_without_lying` relies on.
    """
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        assert read_frame_by_seek(cap, 900, 20) is None
        assert read_frame_by_seek(cap, 20, 20) is None  # off by one
        assert read_frame_by_seek(cap, -1, 20) is None
    finally:
        cap.release()


def test_read_frame_by_seek_refuses_unusable_frame_count(tiny_video):
    """
    A container that reports no usable frame count gets the walk. cv2
    reports INT64_MIN for some truncated files, so this is not academic.
    """
    cap = open_video(tiny_video)
    assert cap is not None
    try:
        assert read_frame_by_seek(cap, 5, 0) is None
        assert read_frame_by_seek(cap, 5, -9223372036854775808) is None
    finally:
        cap.release()
