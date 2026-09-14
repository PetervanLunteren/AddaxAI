"""
Tests for VideoDetectionModel: every video goes through the tracking
script, on the explicit file list the worker's scan produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ml.inference.video_detector import VideoDetectionModel
from app.ml.track_filter import TRACK_FILTER_BY_DOMAIN


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
    monkeypatch.setattr(
        "app.ml.inference.video_detector.resolve_ffmpeg", lambda env: "/env/bin/ffmpeg"
    )
    model_path = tmp_path / "md.pt"
    model_path.write_bytes(b"weights")
    return VideoDetectionModel(model_path, _FakeEnvManager(), env_name="addaxai-base")


def _run(model, tmp_path, output_json, videos, **overrides):
    kwargs = dict(
        video_folder=tmp_path,
        video_files=videos,
        output_json=output_json,
        crops_dir=tmp_path / "frames",
        fps=2.0,
        class_mapping=None,
        track_filter=TRACK_FILTER_BY_DOMAIN["camera_trap"],
    )
    kwargs.update(overrides)
    return model.detect_videos_to_json(**kwargs)


def test_writes_the_file_list_and_runs_the_tracking_script(model, tmp_path, monkeypatch):
    """The script reads an explicit file list (the worker's scan), never
    the folder, so a previous run's output folders cannot be analysed."""
    output_json = tmp_path / "video_results.json"
    commands: list[list[str]] = []

    def fake_stream(command, env, progress_callback, job_id):
        commands.append(command)
        output_json.write_text("{}")
        return 0

    monkeypatch.setattr(model, "_stream_process", fake_stream)
    videos = [tmp_path / "a.mp4", tmp_path / "sub" / "b.MP4"]

    result = _run(model, tmp_path, output_json, videos)

    assert result == output_json
    (command,) = commands
    assert command[2].endswith("tracking_script.py")
    file_list = tmp_path / "video_results_files.json"
    assert json.loads(file_list.read_text()) == [str(p) for p in videos]
    assert str(file_list) in command
    assert command[command.index("--ffmpeg") + 1] == "/env/bin/ffmpeg"
    assert command[command.index("--crops_dir") + 1] == str(tmp_path / "frames")
    assert "--class_mapping" not in command


def test_a_class_mapping_is_written_beside_the_file_list(
    model, tmp_path, monkeypatch
):
    """A detector the megadetector package cannot name (SharkTrack) hands
    its classes over as a file, the package's own ``--class_mapping``
    format. It lands beside the file list so it is cleaned up with the
    run's other artifacts."""
    output_json = tmp_path / "video_results.json"
    commands: list[list[str]] = []

    def fake_stream(command, env, progress_callback, job_id):
        commands.append(command)
        output_json.write_text("{}")
        return 0

    monkeypatch.setattr(model, "_stream_process", fake_stream)

    _run(
        model,
        tmp_path,
        output_json,
        [tmp_path / "a.mp4"],
        class_mapping={"0": "elasmobranch"},
    )

    (command,) = commands
    mapping_path = Path(command[command.index("--class_mapping") + 1])
    assert mapping_path.parent == output_json.parent
    assert json.loads(mapping_path.read_text()) == {"0": "elasmobranch"}


def test_a_failing_script_is_an_error(model, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "_stream_process", lambda *a: 1)
    with pytest.raises(RuntimeError, match="exit code 1"):
        _run(model, tmp_path, tmp_path / "out.json", [tmp_path / "a.mp4"])


def test_without_videos_is_a_configuration_error(model, tmp_path):
    with pytest.raises(ValueError):
        _run(model, tmp_path, tmp_path / "out.json", [])
