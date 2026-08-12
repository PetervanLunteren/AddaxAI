"""
Tests for the access-violation retry in VideoDetectionModel.

OpenCV's FFmpeg backend takes the whole process_video subprocess down
with 0xC0000005 on videos whose pixel format changes mid-stream
(Bushnell MJPEG AVIs: frame 0 is yuvj422p, the rest yuvj420p). The
detector retries once with OPENCV_VIDEOIO_PRIORITY_FFMPEG=0 so cv2
picks MSMF, which decodes those files. See "Mixed pixel format videos"
in DEVELOPERS.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ml.inference.video_detector import (
    _WINDOWS_ACCESS_VIOLATION,
    VideoDetectionModel,
)


class _FakeEnvManager:
    def get_python(self, env_name: str) -> Path:
        return Path("/usr/bin/python3")


@pytest.fixture
def model(tmp_path, monkeypatch):
    # The GPU probe is irrelevant here and spawns a real subprocess on
    # Linux; pin it to "no overrides" so the test is deterministic on
    # every platform.
    monkeypatch.setattr(
        "app.ml.inference.video_detector.cuda_guard_overrides",
        lambda env_manager: {},
    )
    model_path = tmp_path / "md.pt"
    model_path.write_bytes(b"weights")
    return VideoDetectionModel(model_path, _FakeEnvManager())


def test_access_violation_retries_once_with_ffmpeg_deprioritised(
    model, tmp_path, monkeypatch
):
    output_json = tmp_path / "out.json"
    call_envs: list[dict[str, str]] = []

    def fake_stream(command, env, progress_callback, job_id):
        call_envs.append(env)
        if len(call_envs) == 1:
            return _WINDOWS_ACCESS_VIOLATION
        output_json.write_text("{}")
        return 0

    monkeypatch.setattr(model, "_stream_process", fake_stream)

    result = model.detect_videos_to_json(
        video_folder=tmp_path,
        output_json=output_json,
        fps=1.0,
        confidence_threshold=0.005,
    )

    assert result == output_json
    assert len(call_envs) == 2
    assert "OPENCV_VIDEOIO_PRIORITY_FFMPEG" not in call_envs[0]
    assert call_envs[1]["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] == "0"


def test_other_exit_codes_do_not_retry(model, tmp_path, monkeypatch):
    calls: list[dict[str, str]] = []

    def fake_stream(command, env, progress_callback, job_id):
        calls.append(env)
        return 1

    monkeypatch.setattr(model, "_stream_process", fake_stream)

    with pytest.raises(RuntimeError, match="exit code 1"):
        model.detect_videos_to_json(
            video_folder=tmp_path,
            output_json=tmp_path / "out.json",
            fps=1.0,
            confidence_threshold=0.005,
        )
    assert len(calls) == 1


def test_second_access_violation_surfaces_the_error(
    model, tmp_path, monkeypatch
):
    calls: list[dict[str, str]] = []

    def fake_stream(command, env, progress_callback, job_id):
        calls.append(env)
        return _WINDOWS_ACCESS_VIOLATION

    monkeypatch.setattr(model, "_stream_process", fake_stream)

    with pytest.raises(RuntimeError, match=str(_WINDOWS_ACCESS_VIOLATION)):
        model.detect_videos_to_json(
            video_folder=tmp_path,
            output_json=tmp_path / "out.json",
            fps=1.0,
            confidence_threshold=0.005,
        )
    assert len(calls) == 2
