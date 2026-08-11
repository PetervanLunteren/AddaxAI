"""Tests for the GPU guard: decision rule, cached probe API, spawn wiring.

All torch-free. The probe subprocess is faked; the decision function is
pure. See app/ml/gpu_guard.py for the fail-open contract these pin.
"""

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.ml import gpu_guard
from app.ml.gpu_guard import cuda_guard_overrides, decide_hide_cuda

CU128_ARCHES = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def _facts(
    arch: list | None = None,
    devices: list | None = None,
    available: bool = True,
) -> dict:
    return {
        "cuda_available": available,
        "arch_list": CU128_ARCHES if arch is None else arch,
        "devices": [] if devices is None else devices,
    }


def _device(major: int, minor: int, name: str = "GPU") -> dict:
    return {"index": 0, "name": name, "capability": [major, minor]}


# ---------------------------------------------------------------------------
# decide_hide_cuda
# ---------------------------------------------------------------------------


def test_pascal_only_hides():
    message = decide_hide_cuda(
        _facts(devices=[_device(6, 1, "Quadro P4000")])
    )
    assert message is not None
    assert "Quadro P4000" in message
    assert "6.1" in message
    assert "7.0" in message
    assert "CPU" in message


def test_multi_gpu_all_pascal_hides_and_names_both():
    message = decide_hide_cuda(
        _facts(devices=[_device(6, 1, "Quadro P4000"), _device(6, 1, "Quadro P4000")])
    )
    assert message is not None
    assert message.count("Quadro P4000") == 2


def test_mixed_pascal_and_modern_does_not_hide():
    facts = _facts(devices=[_device(6, 1), _device(8, 6)])
    assert decide_hide_cuda(facts) is None


def test_cuda_unavailable_does_not_hide():
    facts = _facts(available=False, devices=[_device(6, 1)])
    assert decide_hide_cuda(facts) is None


def test_empty_arch_list_does_not_hide():
    # mac wheel, CPU wheel, or a driver too old for the CUDA runtime.
    facts = _facts(arch=[], devices=[_device(6, 1)])
    assert decide_hide_cuda(facts) is None


def test_ptx_only_arch_list_does_not_hide():
    # compute_XX entries are PTX, which never runs on an older arch.
    facts = _facts(arch=["compute_120"], devices=[_device(6, 1)])
    assert decide_hide_cuda(facts) is None


def test_no_devices_does_not_hide():
    assert decide_hide_cuda(_facts(devices=[])) is None


def test_capability_equal_to_minimum_is_supported():
    facts = _facts(devices=[_device(7, 0)])
    assert decide_hide_cuda(facts) is None


def test_pascal_on_windows_wheel_is_supported():
    # The real Windows cu128 wheel keeps sm_61 (verified on an RTX 3080 Ti
    # machine, 2026-08-11), unlike Linux. Pascal must NOT be hidden there:
    # the kernels exist and the GPU genuinely works.
    windows_arches = [
        "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120",
    ]
    facts = _facts(arch=windows_arches, devices=[_device(6, 1, "Quadro P4000")])
    assert decide_hide_cuda(facts) is None


@pytest.mark.parametrize(
    "facts",
    [
        None,
        "garbage",
        {},
        {"cuda_available": True},
        _facts(arch="sm_70", devices=[_device(6, 1)]),
        _facts(devices="not-a-list"),
        _facts(devices=[{"index": 0, "name": "GPU", "capability": ["a", "b"]}]),
        _facts(devices=[{"index": 0, "name": "GPU", "capability": [6]}]),
        _facts(devices=[{"index": 0, "name": "GPU"}]),
        _facts(devices=["not-a-dict"]),
    ],
)
def test_malformed_facts_do_not_hide(facts):
    assert decide_hide_cuda(facts) is None


# ---------------------------------------------------------------------------
# cuda_guard_overrides
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_guard(monkeypatch):
    """Fresh memo per test, and a non-darwin platform so results do not
    depend on the dev machine."""
    monkeypatch.setattr(gpu_guard, "_cached_overrides", None)
    monkeypatch.setattr(gpu_guard.platform, "system", lambda: "Linux")


@pytest.fixture
def env_manager():
    return SimpleNamespace(get_python=lambda name: Path("/fake/python"))


class _FakeRun:
    """Callable subprocess.run replacement with a call counter."""

    def __init__(self, returncode=0, stdout="", stderr="", raises=None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raises = raises
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(
            returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def _probe_stdout(facts: dict) -> str:
    return json.dumps(facts) + "\n"


def test_hide_verdict_returns_override(monkeypatch, env_manager):
    fake = _FakeRun(stdout=_probe_stdout(_facts(devices=[_device(6, 1)])))
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)
    assert cuda_guard_overrides(env_manager) == {"CUDA_VISIBLE_DEVICES": "-1"}


def test_supported_gpu_returns_empty(monkeypatch, env_manager):
    fake = _FakeRun(stdout=_probe_stdout(_facts(devices=[_device(8, 6)])))
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)
    assert cuda_guard_overrides(env_manager) == {}


@pytest.mark.parametrize("capability", [(6, 1), (8, 6)])
def test_probe_runs_once_per_process(monkeypatch, env_manager, capability):
    fake = _FakeRun(
        stdout=_probe_stdout(_facts(devices=[_device(*capability)]))
    )
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)
    first = cuda_guard_overrides(env_manager)
    second = cuda_guard_overrides(env_manager)
    assert first == second
    assert fake.calls == 1


def test_nonzero_exit_fails_open_and_caches(monkeypatch, env_manager, caplog):
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "gpu_probe.py", line 39, in main\n'
        "    import torch\n"
        "ModuleNotFoundError: No module named 'torch'"
    )
    fake = _FakeRun(returncode=1, stderr=stderr)
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)
    with caplog.at_level("WARNING"):
        assert cuda_guard_overrides(env_manager) == {}
    assert cuda_guard_overrides(env_manager) == {}
    assert fake.calls == 1
    # The record must stay one line so grep-style log tooling shows the
    # actual error, not just the first traceback line.
    record = next(r for r in caplog.records if "GPU probe" in r.message)
    assert "\n" not in record.message
    assert "No module named 'torch'" in record.message


def test_garbage_stdout_fails_open(monkeypatch, env_manager):
    fake = _FakeRun(stdout="not json at all")
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)
    assert cuda_guard_overrides(env_manager) == {}


def test_last_stdout_line_wins(monkeypatch, env_manager):
    # A stray print before the JSON line must not break parsing.
    stdout = "some stray warning\n" + _probe_stdout(_facts(devices=[_device(6, 1)]))
    fake = _FakeRun(stdout=stdout)
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)
    assert cuda_guard_overrides(env_manager) == {"CUDA_VISIBLE_DEVICES": "-1"}


def test_timeout_fails_open(monkeypatch, env_manager):
    fake = _FakeRun(raises=subprocess.TimeoutExpired(cmd="probe", timeout=120))
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)
    assert cuda_guard_overrides(env_manager) == {}


def test_missing_env_fails_open_without_caching(monkeypatch):
    def _raise(name):
        raise FileNotFoundError(name)

    fake = _FakeRun(stdout=_probe_stdout(_facts(devices=[_device(6, 1)])))
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)

    assert cuda_guard_overrides(SimpleNamespace(get_python=_raise)) == {}
    assert fake.calls == 0

    # After setup installs the env, the next call probes normally.
    working = SimpleNamespace(get_python=lambda name: Path("/fake/python"))
    assert cuda_guard_overrides(working) == {"CUDA_VISIBLE_DEVICES": "-1"}
    assert fake.calls == 1


def test_darwin_short_circuits(monkeypatch):
    monkeypatch.setattr(gpu_guard.platform, "system", lambda: "Darwin")
    fake = _FakeRun()
    monkeypatch.setattr(gpu_guard.subprocess, "run", fake)

    def _fail(name):
        raise AssertionError("get_python must not be called on darwin")

    assert cuda_guard_overrides(SimpleNamespace(get_python=_fail)) == {}
    assert fake.calls == 0


# ---------------------------------------------------------------------------
# Spawn wiring (representative site: MegaDetector images)
# ---------------------------------------------------------------------------


def test_megadetector_spawn_env_carries_override(monkeypatch, tmp_path):
    from app.ml.inference.megadetector import MegaDetectorV1000

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    image = tmp_path / "img.jpg"
    image.write_bytes(b"fake")
    output_path = tmp_path / "results.json"

    env_manager = SimpleNamespace(get_python=lambda name: Path("/fake/python"))
    detector = MegaDetectorV1000(model_path, env_manager)

    captured = {}

    def fake_popen_group(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        # temp output json is the last positional argument.
        Path(cmd[-1]).write_text(json.dumps({"images": []}))
        process = MagicMock()
        process.stdout = io.StringIO("")
        process.returncode = 0
        return process

    monkeypatch.setattr(
        "app.ml.inference.megadetector.cuda_guard_overrides",
        lambda em: {"CUDA_VISIBLE_DEVICES": "-1"},
    )
    monkeypatch.setattr(
        "app.ml.inference.megadetector.popen_group", fake_popen_group
    )

    detector.detect_to_json(
        image_paths=[image],
        deployment_folder=tmp_path,
        confidence_threshold=0.005,
        output_path=output_path,
    )

    # The override must pass THROUGH clean_python_env, not around it.
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"
    assert output_path.exists()
