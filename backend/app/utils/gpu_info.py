"""Graphics card facts for the diagnostics bundle.

Answers "what card is this and how much memory does it have", which is the
question behind every out-of-memory report. The backend process has no
torch, so this asks the driver instead: ``nvidia-smi`` is installed with
every NVIDIA driver, needs no conda environment, and answers in
milliseconds.

Deliberately separate from ``app/ml/inference/gpu_probe.py``. That one asks
whether torch's bundled kernels can run on the card (compute capability
against the wheel's arch list) and has to run inside the torch environment
to answer at all. This one is a hardware question and must work on a
machine where setup never finished. Same subject, different questions.

Best effort throughout: a machine with no GPU, no driver, or a hung
``nvidia-smi`` reports what it can and never raises. A diagnostics bundle
that fails to build is worth less than one with a missing field.
"""

from __future__ import annotations

import platform
import subprocess

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Generous enough for a driver waking a sleeping discrete GPU, short
# enough that a wedged binary cannot hold up the download.
_TIMEOUT_SECONDS = 10

# Asked in one query so a multi-GPU machine returns one line per card.
_NVIDIA_FIELDS = ("name", "memory.total", "driver_version", "compute_cap")


def _run(command: list[str]) -> str | None:
    """stdout of `command`, or None if it is missing, fails or hangs."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        # FileNotFoundError is the ordinary case (no NVIDIA driver), so
        # this stays at debug rather than crying wolf on every Mac.
        logger.debug(f"{command[0]} unavailable: {e}")
        return None
    if result.returncode != 0:
        logger.debug(f"{command[0]} exited {result.returncode}")
        return None
    return result.stdout


def _nvidia_gpus() -> list[dict[str, object]]:
    """One entry per NVIDIA card, or [] when there is no driver."""
    out = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(_NVIDIA_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []

    gpus: list[dict[str, object]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        gpu: dict[str, object] = {"vendor": "NVIDIA", "name": parts[0]}
        # nounits gives MiB as a bare number. Report bytes, so the field
        # needs no unit knowledge to compare against a model's footprint.
        try:
            gpu["memory_total_bytes"] = int(float(parts[1])) * 1024 * 1024
        except ValueError:
            gpu["memory_total_raw"] = parts[1]
        if len(parts) > 2:
            gpu["driver_version"] = parts[2]
        # compute_cap arrived in a later nvidia-smi, and an older one drops
        # the column rather than erroring, so treat it as optional.
        if len(parts) > 3 and parts[3] not in ("", "[N/A]"):
            gpu["compute_capability"] = parts[3]
        gpus.append(gpu)
    return gpus


def _apple_gpu() -> list[dict[str, object]]:
    """The Apple Silicon SoC, whose GPU shares system memory.

    Reported as one entry with the same shape as a discrete card, because
    the reader is asking the same question. `memory_total_bytes` is the
    machine's whole RAM: on unified memory there is no separate VRAM
    figure to give, which `memory_is_unified` says out loud so nobody
    reads it as a dedicated 32 GB card.
    """
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    total = _run(["sysctl", "-n", "hw.memsize"])
    if not chip:
        return []
    gpu: dict[str, object] = {
        "vendor": "Apple",
        "name": chip.strip(),
        "memory_is_unified": True,
    }
    try:
        gpu["memory_total_bytes"] = int((total or "").strip())
    except ValueError:
        pass
    return [gpu]


def collect_gpu_info() -> dict[str, object]:
    """Cards visible to the machine, for the diagnostics bundle.

    An empty `gpus` list is a real answer, not a failure: it means no
    NVIDIA driver responded, so anything that wanted CUDA ran on the CPU.
    """
    gpus = _nvidia_gpus()
    if not gpus and platform.system() == "Darwin":
        gpus = _apple_gpu()
    return {"gpus": gpus}
