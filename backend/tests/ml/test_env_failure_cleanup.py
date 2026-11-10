"""Tests for the cleanup after a failed environment build.

When micromamba fails, the half-built temp folder is removed before the
build error is reported. On Windows the antivirus is often still scanning
the files micromamba just wrote, so that delete can fail with WinError 5
on a single `.pyd`. That cleanup error used to escape and replace the
build error on the setup screen, and the ERROR summary of micromamba's
output was never written (beta report, McAfee, 2026-09-10). These tests
pin that the user sees micromamba's own error, that the log holds it,
and that a locked temp folder never breaks the next build.
"""

import logging
from pathlib import Path
from typing import Any

import pytest

from app.ml import environment_manager
from app.ml.environment_manager import EnvironmentManager
from app.utils.subprocess_runner import StreamedResult

YAML = """name: env-probe
channels:
  - conda-forge
dependencies:
  - python=3.11
"""

TEMP_NAME = ".probe.tmp"
BUILD_TAIL = [
    "info     libmamba Linking python-3.11.9",
    "ERROR: Could not install packages due to an OSError",
    "critical libmamba pip failed to install packages",
]


def prepare_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int,
    lock_temp: bool,
) -> tuple[EnvironmentManager, Path, Path, dict[str, Any]]:
    """Set up a build whose micromamba step exits with `returncode`
    after writing one file into the temp env. With `lock_temp`, every
    attempt to remove a temp env is refused the way an antivirus-held
    file refuses it on Windows.

    Returns the manager, the env path, the yaml path, and a record of
    the temp paths micromamba was asked to build into.
    """
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text(YAML)

    # A file that already exists, so `_ensure_runtime_dirs` does not try
    # to download the real micromamba binary during the test.
    micromamba = tmp_path / "micromamba"
    micromamba.write_text("")

    env_path = tmp_path / "envs" / "env-probe"
    record: dict[str, Any] = {"build_paths": []}

    def fake_stream(cmd: list[str], **kwargs: Any) -> StreamedResult:
        build_path = Path(cmd[cmd.index("-p") + 1])
        record["build_paths"].append(build_path)
        (build_path / "DLLs").mkdir(parents=True, exist_ok=True)
        (build_path / "DLLs" / "_ctypes.pyd").write_bytes(b"")
        return StreamedResult(
            returncode=returncode,
            last_line=BUILD_TAIL[-1],
            output_tail=list(BUILD_TAIL),
        )

    monkeypatch.setattr(environment_manager, "stream_with_tail", fake_stream)

    if lock_temp:
        def refused_rmtree(self: EnvironmentManager, path: Path) -> None:
            raise PermissionError(
                13, "Access is denied", str(path / "DLLs" / "_ctypes.pyd")
            )

        monkeypatch.setattr(EnvironmentManager, "_safe_rmtree", refused_rmtree)

    mgr = EnvironmentManager(envs_dir=tmp_path / "envs", micromamba_path=micromamba)
    return mgr, env_path, yaml_path, record


def test_a_locked_temp_env_does_not_hide_the_build_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """micromamba fails and the cleanup is refused: the error the user
    gets is micromamba's, the log holds its tail at ERROR, and the locked
    folder is left where it is."""
    mgr, env_path, yaml_path, _ = prepare_build(
        tmp_path, monkeypatch, returncode=1, lock_temp=True
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as exc:
            mgr._create_env("probe", env_path, yaml_path)

    message = str(exc.value)
    assert "micromamba create failed (exit 1)" in message
    assert BUILD_TAIL[-1] in message
    assert "_ctypes.pyd" not in message  # the cleanup error stays out of it

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("micromamba create failed" in r.getMessage() for r in errors)
    assert any(
        "Could not remove temporary environment" in r.getMessage()
        for r in caplog.records
    )

    assert (env_path.parent / TEMP_NAME).is_dir()
    assert not env_path.exists()


def test_a_failed_build_removes_its_temp_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary case: micromamba fails, the temp env is gone."""
    mgr, env_path, yaml_path, _ = prepare_build(
        tmp_path, monkeypatch, returncode=1, lock_temp=False
    )

    with pytest.raises(RuntimeError) as exc:
        mgr._create_env("probe", env_path, yaml_path)

    assert BUILD_TAIL[-1] in str(exc.value)
    assert not (env_path.parent / TEMP_NAME).exists()
    assert not env_path.exists()


def test_a_locked_stale_temp_env_moves_the_build_to_a_fresh_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A temp folder left locked by an earlier run cannot be removed, so
    the new build goes into a fresh path instead of the leftover."""
    mgr, env_path, yaml_path, record = prepare_build(
        tmp_path, monkeypatch, returncode=0, lock_temp=True
    )
    stale = env_path.parent / TEMP_NAME
    stale.mkdir(parents=True)
    (stale / "leftover").write_text("")

    mgr._create_env("probe", env_path, yaml_path)

    (build_path,) = record["build_paths"]
    assert build_path != stale
    assert build_path.name.startswith(".probe.tmp-")
    assert env_path.is_dir()
    assert (stale / "leftover").exists()
