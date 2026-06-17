"""Tests for on-demand video filmstrip decoding (filmstrip_service)."""

from __future__ import annotations

import base64
import io

import pytest

cv2 = pytest.importorskip("cv2")
from PIL import Image  # noqa: E402

from app.ml.inference.video_iter import open_video, sample_indices  # noqa: E402
from app.services.filmstrip_service import (  # noqa: E402
    FILMSTRIP_FRAME_COUNT,
    FILMSTRIP_MAX_WIDTH,
    build_filmstrip,
)


def _reported_total(path) -> int:
    """Frame count as cv2 reports it — the same number the service samples."""
    cap = open_video(path)
    assert cap is not None
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _decode(data_uri: str) -> Image.Image:
    assert data_uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    return Image.open(io.BytesIO(raw))


def test_filmstrip_evenly_spaced_and_decodable(tmp_path, make_video):
    video = tmp_path / "clip.mp4"
    # Wider than the cap so the downscale path is exercised.
    make_video(video, total_frames=20, fps=10, size=(640, 480))

    frames = build_filmstrip(str(video), 10.0)

    expected = sample_indices(_reported_total(video), FILMSTRIP_FRAME_COUNT)
    assert [f["frame_number"] for f in frames] == expected
    # time = frame_number / fps
    assert frames[1]["time_seconds"] == pytest.approx(expected[1] / 10.0)
    # each frame is a decodable JPEG, downscaled to the width cap
    for f in frames:
        img = _decode(f["image"])
        assert img.format == "JPEG"
        assert img.width == FILMSTRIP_MAX_WIDTH


def test_filmstrip_short_video_returns_all_frames(tmp_path, make_video):
    video = tmp_path / "short.mp4"
    make_video(video, total_frames=4, fps=10)

    frames = build_filmstrip(str(video), 10.0)

    assert [f["frame_number"] for f in frames] == sample_indices(
        _reported_total(video), FILMSTRIP_FRAME_COUNT
    )
    assert len(frames) <= 4


def test_filmstrip_without_frame_rate_has_null_time(tmp_path, make_video):
    video = tmp_path / "nofps.mp4"
    make_video(video, total_frames=20, fps=10)

    frames = build_filmstrip(str(video), None)

    assert frames
    assert all(f["time_seconds"] is None for f in frames)


def test_filmstrip_unreadable_video_returns_empty(tmp_path):
    assert build_filmstrip(str(tmp_path / "missing.mp4"), 10.0) == ()
