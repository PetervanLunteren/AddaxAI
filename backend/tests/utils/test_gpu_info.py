"""Tests for the diagnostics bundle's graphics card facts.

Every branch here is a machine we cannot run CI on, so each one fakes the
tool's output rather than the hardware. The point of the module is that a
bundle still builds when the tool is missing or wedged, so most of these
are about the failure paths.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from app.utils import gpu_info


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=""
    )


def test_single_nvidia_card_reports_bytes_not_mib():
    """MiB is what nvidia-smi speaks; the bundle should speak bytes."""
    out = "NVIDIA GeForce RTX 4070, 12282, 566.36, 8.9\n"
    with patch("subprocess.run", return_value=_completed(out)):
        gpus = gpu_info.collect_gpu_info()["gpus"]

    assert len(gpus) == 1
    assert gpus[0] == {
        "vendor": "NVIDIA",
        "name": "NVIDIA GeForce RTX 4070",
        "memory_total_bytes": 12282 * 1024 * 1024,
        "driver_version": "566.36",
        "compute_capability": "8.9",
    }


def test_multiple_cards_each_get_an_entry():
    out = "NVIDIA RTX A6000, 49140, 550.90, 8.6\nNVIDIA RTX A6000, 49140, 550.90, 8.6\n"
    with patch("subprocess.run", return_value=_completed(out)):
        gpus = gpu_info.collect_gpu_info()["gpus"]

    assert len(gpus) == 2
    assert {g["name"] for g in gpus} == {"NVIDIA RTX A6000"}


def test_older_nvidia_smi_without_compute_cap_column():
    """The column is newer than the tool; its absence is not an error."""
    out = "Quadro P2000, 5058, 452.39\n"
    with patch("subprocess.run", return_value=_completed(out)):
        gpus = gpu_info.collect_gpu_info()["gpus"]

    assert gpus[0]["name"] == "Quadro P2000"
    assert gpus[0]["driver_version"] == "452.39"
    assert "compute_capability" not in gpus[0]


def test_no_driver_reports_an_empty_list_rather_than_failing():
    """The ordinary case on a machine with no NVIDIA card."""
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi")):
        with patch("platform.system", return_value="Windows"):
            assert gpu_info.collect_gpu_info() == {"gpus": []}


def test_a_wedged_nvidia_smi_cannot_hold_up_the_bundle():
    """A hung driver query must time out, not block the download."""
    timeout = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)
    with patch("subprocess.run", side_effect=timeout):
        with patch("platform.system", return_value="Linux"):
            assert gpu_info.collect_gpu_info() == {"gpus": []}


def test_nonzero_exit_is_treated_as_no_answer():
    with patch("subprocess.run", return_value=_completed("", returncode=9)):
        with patch("platform.system", return_value="Linux"):
            assert gpu_info.collect_gpu_info() == {"gpus": []}


def test_unparseable_memory_is_kept_raw_rather_than_dropped():
    """Better to hand a human the odd string than to silently lose it."""
    out = "NVIDIA T400, [N/A], 535.104, 7.5\n"
    with patch("subprocess.run", return_value=_completed(out)):
        gpus = gpu_info.collect_gpu_info()["gpus"]

    assert "memory_total_bytes" not in gpus[0]
    assert gpus[0]["memory_total_raw"] == "[N/A]"


def test_apple_silicon_falls_back_to_the_soc_and_says_memory_is_shared():
    def fake_run(command, **kwargs):
        if command[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        if command[-1] == "machdep.cpu.brand_string":
            return _completed("Apple M2 Pro\n")
        return _completed("34359738368\n")

    with patch("subprocess.run", side_effect=fake_run):
        with patch("platform.system", return_value="Darwin"):
            gpus = gpu_info.collect_gpu_info()["gpus"]

    assert gpus == [
        {
            "vendor": "Apple",
            "name": "Apple M2 Pro",
            "memory_is_unified": True,
            "memory_total_bytes": 34359738368,
        }
    ]


def test_apple_branch_is_not_used_when_an_nvidia_card_answered():
    """A Linux box with a card must not be asked about sysctl."""
    out = "NVIDIA GeForce RTX 3060, 12288, 535.104, 8.6\n"
    with patch("subprocess.run", return_value=_completed(out)):
        with patch("platform.system", return_value="Darwin"):
            gpus = gpu_info.collect_gpu_info()["gpus"]

    assert len(gpus) == 1
    assert gpus[0]["vendor"] == "NVIDIA"
