"""Tests for the env boot probe in app.ml.environment_manager.

The probe decides whether an installed env is trusted or wiped and
rebuilt, so both directions matter: a pruned stdlib must read as
broken, and a slow-to-start interpreter (antivirus scanning) must NOT,
because a False verdict lets get_or_create_env delete the env.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from app.ml.environment_manager import (
    _BOOT_PROBE_SCRIPT,
    EnvironmentManager,
)


@pytest.fixture
def env_manager(tmp_path: Path) -> EnvironmentManager:
    """
    EnvironmentManager pointed at a temp envs dir. _ensure_runtime_dirs
    is patched out to avoid the real micromamba download.
    """
    with patch.object(
        EnvironmentManager, "_ensure_runtime_dirs", lambda self: None
    ):
        return EnvironmentManager(envs_dir=tmp_path / "envs")


@pytest.fixture
def env_path(env_manager: EnvironmentManager) -> Path:
    """A fake env directory whose python binary exists on disk."""
    path = env_manager.envs_dir / "env-addaxai-base"
    python_path = env_manager._get_python_path(path)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"")
    return path


class _Result:
    def __init__(self, returncode: int, stderr: bytes = b""):
        self.returncode = returncode
        self.stderr = stderr


def test_probe_imports_stdlib_extension_modules() -> None:
    """
    The probe must import the stdlib C extension modules, not just
    boot the interpreter. An env whose DLLs were quarantined by
    antivirus boots fine on `import sys` and then fails every
    analysis (field case 2026-08: select.pyd and unicodedata.pyd
    missing, every check green, every job dead).
    """
    for module in ("encodings", "select", "unicodedata", "ssl"):
        assert module in _BOOT_PROBE_SCRIPT


def test_missing_python_binary_is_invalid(
    env_manager: EnvironmentManager,
) -> None:
    path = env_manager.envs_dir / "env-addaxai-base"
    path.mkdir(parents=True)
    assert env_manager._validate_env(path) is False


def test_probe_success_is_valid(
    env_manager: EnvironmentManager, env_path: Path
) -> None:
    with patch(
        "app.ml.environment_manager.subprocess.run",
        return_value=_Result(0),
    ) as run:
        assert env_manager._validate_env(env_path) is True
    assert _BOOT_PROBE_SCRIPT in run.call_args.args[0]


def test_probe_nonzero_exit_is_invalid(
    env_manager: EnvironmentManager, env_path: Path
) -> None:
    """A completed probe with a non-zero exit is a provably broken env."""
    with patch(
        "app.ml.environment_manager.subprocess.run",
        return_value=_Result(1, stderr=b"ModuleNotFoundError: select"),
    ):
        assert env_manager._validate_env(env_path) is False


def test_probe_timeout_is_valid(
    env_manager: EnvironmentManager, env_path: Path
) -> None:
    """
    A timeout means the machine is slow (antivirus scanning python.exe
    at launch), not that the env is broken: a broken env fails in
    milliseconds. Timeout must not return False, or get_or_create_env
    wipes and rebuilds a healthy env.
    """
    with patch(
        "app.ml.environment_manager.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="python", timeout=10),
    ):
        assert env_manager._validate_env(env_path) is True


def test_probe_oserror_is_invalid(
    env_manager: EnvironmentManager, env_path: Path
) -> None:
    """A python.exe that cannot even launch is a broken env."""
    with patch(
        "app.ml.environment_manager.subprocess.run",
        side_effect=OSError("exec format error"),
    ):
        assert env_manager._validate_env(env_path) is False
