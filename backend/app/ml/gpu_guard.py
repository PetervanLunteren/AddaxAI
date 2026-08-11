"""Force CPU when no visible NVIDIA GPU is supported by the bundled torch.

The cu128 torch wheels shipped in the ML envs only contain kernels for
compute capability 7.0 to 12.0. On an older GPU (Pascal and before) with
a recent driver, ``torch.cuda.is_available()`` still returns True, the
detector picks cuda:0, every inference batch fails with "no kernel image
is available", and MegaDetector swallows those per batch as image-level
failures with exit code 0 — a silently empty run. See DEVELOPERS.md.

The fix: probe the raw CUDA facts once with the env's own torch
(``app/ml/inference/gpu_probe.py``), decide here, and when no visible
GPU can run the bundled kernels, spawn the ML subprocesses with
``CUDA_VISIBLE_DEVICES=-1``. Both torch and TensorFlow then take their
normal, well-tested CPU path. Every failure path returns no override,
which is exactly today's behavior (fail open).
"""

import json
import platform
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.logging_config import get_logger
from app.utils.subprocess_env import clean_python_env

if TYPE_CHECKING:
    from app.ml.environment_manager import EnvironmentManager

logger = get_logger(__name__)

_PROBE_SCRIPT = Path(__file__).parent / "inference" / "gpu_probe.py"

# torch import on a cold Windows disk behind Defender is the slow path;
# a false timeout costs us the guard, a real hang still fails open.
_PROBE_TIMEOUT_S = 120

# Real-kernel entries only. compute_XX PTX entries are deliberately
# ignored: PTX embedded for a newer arch can never JIT for an older one,
# so they add no support.
_SM_RE = re.compile(r"^sm_(\d+)$")

# None = not probed yet. Consumed only via **, never mutated.
_cached_overrides: dict[str, str] | None = None


def decide_hide_cuda(facts: object) -> str | None:
    """Warning message when every visible GPU is below the minimum
    compute capability compiled into the bundled torch; None otherwise.

    Missing or malformed data always returns None (fail open).
    """
    try:
        if not isinstance(facts, dict) or facts.get("cuda_available") is not True:
            return None

        sm_nums = [
            int(m.group(1))
            for entry in facts["arch_list"]
            if isinstance(entry, str) and (m := _SM_RE.match(entry))
        ]
        if not sm_nums:
            return None
        min_sm = min(sm_nums)

        devices = facts["devices"]
        if not isinstance(devices, list) or not devices:
            return None

        names = []
        for device in devices:
            major, minor = device["capability"]
            if int(major) * 10 + int(minor) >= min_sm:
                return None
            names.append(f"{device['name']} (compute capability {int(major)}.{int(minor)})")

        return (
            f"CUDA GPU(s) too old for the bundled PyTorch: {', '.join(names)}. "
            f"Minimum supported is {min_sm / 10:.1f}. Analysis will run on the CPU."
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def cuda_guard_overrides(env_manager: "EnvironmentManager") -> dict[str, str]:
    """Env overrides for ML inference subprocesses.

    Returns {} normally, or {"CUDA_VISIBLE_DEVICES": "-1"} when no
    visible GPU can run the bundled CUDA kernels. Probes at most once
    per backend process.
    """
    global _cached_overrides

    if platform.system() == "Darwin":
        return {}

    if _cached_overrides is not None:
        return _cached_overrides

    # Not cached: this only happens mid-setup, retrying is two
    # Path.exists() calls, and caching would leave the whole backend
    # session unguarded after setup completes.
    try:
        python_path = env_manager.get_python("env-addaxai-base")
    except FileNotFoundError:
        logger.warning("GPU probe skipped: env-addaxai-base not installed yet")
        return {}

    try:
        result = subprocess.run(
            [str(python_path), str(_PROBE_SCRIPT)],
            env=clean_python_env(),
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning(f"GPU probe failed to run: {e}")
        _cached_overrides = {}
        return _cached_overrides

    if result.returncode != 0:
        # Joined with " | " so the record stays one line: grep and other
        # line-based log tooling would otherwise show only the first,
        # least useful line of the traceback.
        stderr_tail = " | ".join(result.stderr.strip().splitlines()[-5:])
        logger.warning(
            f"GPU probe exited with code {result.returncode}: {stderr_tail}"
        )
        _cached_overrides = {}
        return _cached_overrides

    try:
        facts = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        logger.warning(f"GPU probe output not parseable: {result.stdout[:200]!r}")
        _cached_overrides = {}
        return _cached_overrides

    message = decide_hide_cuda(facts)
    if message:
        logger.warning(message)
        _cached_overrides = {"CUDA_VISIBLE_DEVICES": "-1"}
    else:
        logger.info("GPU probe: CUDA supported or absent, no override")
        _cached_overrides = {}
    return _cached_overrides
