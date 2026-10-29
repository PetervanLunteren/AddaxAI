"""`detect_to_json` reads MegaDetector's output as UTF-8 whatever the
locale, and tells the child to write UTF-8. Left to the locale, a cp932
parent reading a child that inherited PYTHONIOENCODING=utf-8 raised
UnicodeDecodeError on the first non-ASCII line and failed the run.
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
    def __init__(self, cmd):
        file_list = json.loads(Path(cmd[-2]).read_text())
        results = [{"file": f, "detections": []} for f in file_list]
        Path(cmd[-1]).write_text(json.dumps({"images": results}))
        self.stdout = io.StringIO("")
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
    return MegaDetectorV1000(model_path, _FakeEnvManager())


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
        env=md_module.clean_python_env(PYTHONIOENCODING="utf-8"),
    )
    lines = [line.strip() for line in process.stdout]
    process.wait()
    assert lines == ["カメラ 100%|██████████| 1/1"]
