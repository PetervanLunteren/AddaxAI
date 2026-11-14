"""`detect_to_json` reads MegaDetector's output as UTF-8 whatever the
locale, and tells the child to write UTF-8. Left to the locale, a cp932
parent reading a child that inherited PYTHONIOENCODING=utf-8 raised
UnicodeDecodeError on the first non-ASCII line and failed the run.

With the pipe in UTF-8, tqdm in the child draws Unicode bar characters.
Those must only go to the log: echoing them to the backend's own stdout,
a cp1252 pipe under Electron on Windows, raised UnicodeEncodeError and
failed every image run in 7.18.0 and 7.18.1.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.ml.inference import megadetector as md_module
from app.ml.inference.megadetector import MegaDetectorV1000


class _FakeEnvManager:
    def get_python(self, env_name: str) -> Path:
        return Path("/usr/bin/python3")


class _FakeProcess:
    def __init__(self, cmd, stdout_text: str = ""):
        file_list = json.loads(Path(cmd[-2]).read_text())
        results = [{"file": f, "detections": []} for f in file_list]
        Path(cmd[-1]).write_text(json.dumps({"images": results}))
        self.stdout = io.StringIO(stdout_text)
        self.returncode = 0

    def wait(self) -> int:
        return self.returncode


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        md_module, "cuda_guard_overrides", lambda env_manager: {"CUDA_GUARD": "1"}
    )
    model_path = tmp_path / "md.pt"
    model_path.write_bytes(b"weights")
    return MegaDetectorV1000(model_path, _FakeEnvManager(), env_name="addaxai-base")


def test_the_pipe_is_utf8_on_both_ends(model, tmp_path, monkeypatch):
    folder = tmp_path / "deployment"
    folder.mkdir()
    (folder / "IMG_0.jpg").write_bytes(b"jpg")
    monkeypatch.setenv("PYTHONIOENCODING", "cp932")
    captured: dict = {}

    def fake_popen(cmd, **popen_kwargs):
        captured.update(popen_kwargs)
        return _FakeProcess(cmd)

    monkeypatch.setattr(md_module, "popen_group", fake_popen)

    model.detect_to_json(
        image_paths=[folder / "IMG_0.jpg"],
        deployment_folder=folder,
        confidence_threshold=0.01,
        output_path=folder / "detection_image.json",
    )

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    # The CUDA guard overrides still reach the child.
    assert captured["env"]["CUDA_GUARD"] == "1"


def test_a_unicode_progress_bar_does_not_fail_the_run_on_a_cp1252_console(
    model, tmp_path, monkeypatch, caplog
):
    folder = tmp_path / "deployment"
    folder.mkdir()
    (folder / "IMG_0.jpg").write_bytes(b"jpg")
    bar = " 1%|\u258f         | 5/510 [00:00<00:42, 11.94it/s]"
    monkeypatch.setattr(
        md_module,
        "popen_group",
        lambda cmd, **_kwargs: _FakeProcess(cmd, stdout_text=bar + "\n"),
    )
    # The backend's stdout under Electron on Windows: a pipe in the locale
    # code page, which has no block characters.
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    with caplog.at_level("INFO", logger=md_module.logger.name):
        model.detect_to_json(
            image_paths=[folder / "IMG_0.jpg"],
            deployment_folder=folder,
            confidence_threshold=0.01,
            output_path=folder / "detection_image.json",
        )

    assert f"MegaDetector: {bar.strip()}" in caplog.text


def test_a_failing_run_says_why_in_the_error(model, tmp_path, monkeypatch):
    """The screen shows only the error, so it has to carry the cause: the
    last line the script printed, which for a Python crash is the line of
    the traceback that names the exception. It used to be a bare "return
    code 1" (forum report 2026-11-11)."""
    folder = tmp_path / "deployment"
    folder.mkdir()
    (folder / "IMG_0.jpg").write_bytes(b"jpg")
    output = (
        "Traceback (most recent call last):\n"
        '  File "run_detector_batch.py", line 1, in <module>\n'
        "RuntimeError: CUDA out of memory\n"
        "\n"
    )

    def failing_popen(cmd, **_kwargs):
        process = _FakeProcess(cmd, stdout_text=output)
        process.returncode = 1
        return process

    monkeypatch.setattr(md_module, "popen_group", failing_popen)

    with pytest.raises(RuntimeError) as excinfo:
        model.detect_to_json(
            image_paths=[folder / "IMG_0.jpg"],
            deployment_folder=folder,
            confidence_threshold=0.01,
            output_path=folder / "detection_image.json",
        )

    assert str(excinfo.value) == (
        "MegaDetector execution failed: exited with code 1: "
        "RuntimeError: CUDA out of memory"
    )


def test_non_ascii_output_survives_a_real_pipe():
    """The same Popen settings, round-tripped through a real child."""
    process = subprocess.Popen(
        [sys.executable, "-c", "print('カメラ 100%|██████████| 1/1')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=md_module.clean_python_env(),
    )
    lines = [line.strip() for line in process.stdout]
    process.wait()
    assert lines == ["カメラ 100%|██████████| 1/1"]
