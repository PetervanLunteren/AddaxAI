"""Tests for the retry around the final environment rename.

On Windows, renaming the finished temp environment into place fails with
WinError 5 while another process holds a handle on any file inside, which
right after a build is almost always the antivirus scanning the fresh
files. One refused rename used to throw away a 10-30 minute build
(observed on an MNHN install, 2026-08-19). `_rename_with_retries` waits
out such transient locks; these tests simulate them, since the race
cannot be reproduced on the POSIX machines that run the suite.
"""

from pathlib import Path
from typing import Any

import pytest

from app.ml import environment_manager
from app.ml.environment_manager import _RENAME_WAITS, EnvironmentManager
from app.utils.subprocess_runner import StreamedResult

YAML = """name: env-probe
channels:
  - conda-forge
dependencies:
  - python=3.11
"""

TEMP_NAME = ".probe.tmp"


def prepare_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rename_failures: int,
    error: OSError | None = None,
) -> tuple[EnvironmentManager, Path, Path, dict[str, Any]]:
    """Set up a build whose micromamba step succeeds instantly and whose
    final rename is refused `rename_failures` times.

    Returns the manager, the env path, the yaml path, and a record dict
    with the rename attempts and the sleeps taken, readable whether or
    not the build ultimately raises.
    """
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text(YAML)

    # A file that already exists, so `_ensure_runtime_dirs` does not try
    # to download the real micromamba binary during the test.
    micromamba = tmp_path / "micromamba"
    micromamba.write_text("")

    env_path = tmp_path / "envs" / "env-probe"

    def fake_stream(cmd: list[str], **kwargs: Any) -> StreamedResult:
        # A successful build leaves the environment in the temp dir;
        # that is what the rename under test moves into place.
        (env_path.parent / TEMP_NAME).mkdir(parents=True, exist_ok=True)
        return StreamedResult(returncode=0, last_line="done", output_tail=[])

    monkeypatch.setattr(environment_manager, "stream_with_tail", fake_stream)

    record: dict[str, Any] = {"attempts": 0, "slept": []}
    monkeypatch.setattr(environment_manager.time, "sleep", record["slept"].append)

    refusal = error or PermissionError(13, "Access is denied", str(env_path))
    original_rename = Path.rename

    def flaky_rename(self: Path, target: Any) -> Any:
        # Only the move under test is affected; every other rename in
        # the process is delegated untouched.
        if self.name != TEMP_NAME:
            return original_rename(self, target)
        record["attempts"] += 1
        if record["attempts"] <= rename_failures:
            raise refusal
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    mgr = EnvironmentManager(envs_dir=tmp_path / "envs", micromamba_path=micromamba)
    return mgr, env_path, yaml_path, record


def test_a_transient_lock_does_not_fail_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two refusals, then the handle is released: the env lands in place."""
    mgr, env_path, yaml_path, record = prepare_build(
        tmp_path, monkeypatch, rename_failures=2
    )

    mgr._create_env("probe", env_path, yaml_path)

    assert env_path.is_dir()
    assert not (env_path.parent / TEMP_NAME).exists()
    assert record["attempts"] == 3
    assert record["slept"] == list(_RENAME_WAITS[:2])


def test_a_persistent_lock_names_the_antivirus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every attempt refused: the error tells the user what to do, the
    whole wait budget was spent first, and the cleanup still removes the
    temp dir."""
    mgr, env_path, yaml_path, record = prepare_build(
        tmp_path, monkeypatch, rename_failures=len(_RENAME_WAITS) + 1
    )

    with pytest.raises(RuntimeError) as exc:
        mgr._create_env("probe", env_path, yaml_path)

    message = str(exc.value)
    assert "antivirus" in message
    assert "Access is denied" in message  # the original OS error survives

    assert record["attempts"] == len(_RENAME_WAITS) + 1
    assert record["slept"] == list(_RENAME_WAITS)  # gave up early would show here

    assert not (env_path.parent / TEMP_NAME).exists()
    assert not env_path.exists()


def test_any_other_error_fails_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry is for held handles only. A different error, here the
    target already existing (WinError 183), must not burn 30 seconds."""
    mgr, env_path, yaml_path, record = prepare_build(
        tmp_path,
        monkeypatch,
        rename_failures=1,
        error=FileExistsError(
            17, "Cannot create a file when that file already exists"
        ),
    )

    with pytest.raises(RuntimeError) as exc:
        mgr._create_env("probe", env_path, yaml_path)

    assert "already exists" in str(exc.value)
    assert "antivirus" not in str(exc.value)
    assert record["slept"] == []
