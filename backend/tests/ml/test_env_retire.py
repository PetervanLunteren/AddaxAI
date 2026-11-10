"""Tests for replacing an installed environment (`retire_env`).

An env used to be deleted in place. On Windows that delete stopped at the
first locked file, and what was left still booted: a beta tester's
"Update now" removed `bin\\` with exiftool, stopped at a locked
`DLLs\\select.pyd`, and every check then called the env healthy
(2026-11-03). `retire_env` renames the env aside first, so an env is
either replaced whole or left exactly as it was. These tests pin both
outcomes, the leftover sweep, and that the build path uses it.
"""

import logging
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

ENV_FILES = ("python.exe", "bin/exiftool.bat", "DLLs/select.pyd", "Lib/os.py")


def make_manager(tmp_path: Path) -> EnvironmentManager:
    # A file that already exists, so the constructor does not try to
    # download the real micromamba binary.
    micromamba = tmp_path / "micromamba"
    micromamba.write_text("")
    return EnvironmentManager(envs_dir=tmp_path / "envs", micromamba_path=micromamba)


def make_env(mgr: EnvironmentManager) -> Path:
    env_path = mgr.envs_dir / "env-probe"
    for rel in ENV_FILES:
        f = env_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("")
    return env_path


def old_copies(mgr: EnvironmentManager) -> list[Path]:
    return sorted(mgr.envs_dir.glob(".env-probe.old-*"))


def test_retire_moves_the_env_aside_and_deletes_it(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    env_path = make_env(mgr)

    mgr.retire_env(env_path)

    assert not env_path.exists()
    assert old_copies(mgr) == []


def test_a_refused_move_leaves_the_env_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every rename refused: the user gets an error that says what failed
    and what to do, the full wait budget was spent, and not one file of
    the env is gone."""
    mgr = make_manager(tmp_path)
    env_path = make_env(mgr)

    slept: list[float] = []
    monkeypatch.setattr(environment_manager.time, "sleep", slept.append)

    def refused(self: Path, target: Any) -> Any:
        raise PermissionError(13, "Access is denied", str(self))

    monkeypatch.setattr(Path, "rename", refused)

    with pytest.raises(RuntimeError) as exc:
        mgr.retire_env(env_path)

    message = str(exc.value)
    assert "move the old environment aside" in message
    assert "running analysis" in message
    assert "Access is denied" in message
    assert slept == list(_RENAME_WAITS)

    for rel in ENV_FILES:
        assert (env_path / rel).is_file(), rel
    assert old_copies(mgr) == []


def test_an_old_copy_that_cannot_be_deleted_is_left_with_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The move worked but the delete is refused (antivirus still
    scanning): no error, because the env is out of the way, and the copy
    waits for the sweep at the next build."""
    mgr = make_manager(tmp_path)
    env_path = make_env(mgr)
    monkeypatch.setattr(environment_manager, "safe_rmtree", lambda path: False)

    with caplog.at_level(logging.WARNING):
        mgr.retire_env(env_path)

    assert not env_path.exists()
    (old,) = old_copies(mgr)
    assert (old / "DLLs" / "select.pyd").is_file()
    assert any("Old environment left at" in r.getMessage() for r in caplog.records)


def test_get_or_create_env_replaces_an_invalid_env_by_moving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rebuild path for an env that fails validation goes through
    retire_env, never through a delete in place."""
    mgr = make_manager(tmp_path)
    env_path = make_env(mgr)
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text(YAML)

    retired: list[Path] = []
    monkeypatch.setattr(EnvironmentManager, "_validate_env", lambda self, p: False)
    monkeypatch.setattr(EnvironmentManager, "get_env_yaml_path", lambda self, n: yaml_path)
    monkeypatch.setattr(EnvironmentManager, "retire_env", lambda self, p: retired.append(p))
    monkeypatch.setattr(
        EnvironmentManager,
        "_safe_rmtree",
        lambda self, p: pytest.fail("an env must not be deleted in place"),
    )
    monkeypatch.setattr(EnvironmentManager, "_create_env", lambda self, *a, **k: None)

    manifest: Any = type("M", (), {"env": "probe"})()
    mgr.get_or_create_env(manifest)

    assert retired == [env_path]


def test_a_build_sweeps_old_copies_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old copies left by an earlier retire are removed before the next
    build, and the build itself goes ahead."""
    mgr = make_manager(tmp_path)
    env_path = mgr.envs_dir / "env-probe"
    leftover = mgr.envs_dir / ".env-probe.old-1234abcd"
    (leftover / "DLLs").mkdir(parents=True)
    (leftover / "DLLs" / "select.pyd").write_text("")
    # An old copy of another env is not this build's business.
    other = mgr.envs_dir / ".env-other.old-1234abcd"
    other.mkdir()

    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text(YAML)

    def fake_stream(cmd: list[str], **kwargs: Any) -> StreamedResult:
        Path(cmd[cmd.index("-p") + 1]).mkdir(parents=True, exist_ok=True)
        return StreamedResult(returncode=0, last_line="done", output_tail=[])

    monkeypatch.setattr(environment_manager, "stream_with_tail", fake_stream)

    mgr._create_env("probe", env_path, yaml_path)

    assert env_path.is_dir()
    assert not leftover.exists()
    assert other.is_dir()
