"""Tests for the one retry after micromamba itself crashes.

micromamba 2.9.0 dies with a memory fault (0xC0000005 on Windows, SIGSEGV
elsewhere) and no output when it parses a malformed shard index, the
compact package index it fetches first (mamba issue #4419). Users on
7.11.0 hit it at the first setup step in October 2026. Without shards
micromamba downloads the full repodata.json and parses it with different
code, so one retry that way gets past a bad index, wherever it came from.
"""

import signal
from pathlib import Path
from typing import Any

import pytest

from app.core.job_cancellation import JobCancelledError
from app.ml import environment_manager
from app.ml.environment_manager import EnvironmentManager
from app.utils.subprocess_runner import WINDOWS_ACCESS_VIOLATION, StreamedResult

YAML = """name: env-probe
channels:
  - conda-forge
dependencies:
  - python=3.11
"""


def build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncodes: list[int],
    job_id: str | None = None,
) -> list[dict[str, str]]:
    """Run `_create_env` against a micromamba that exits with `returncodes`
    in turn. Returns the environment each run was given."""
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text(YAML)
    micromamba = tmp_path / "micromamba"
    micromamba.write_text("")
    runs: list[dict[str, str]] = []

    def fake_stream(cmd: list[str], **kwargs: Any) -> StreamedResult:
        runs.append(dict(kwargs["env"]))
        code = returncodes[len(runs) - 1]
        if code == 0:
            # A successful create leaves the env at its -p path.
            Path(cmd[cmd.index("-p") + 1]).mkdir(parents=True)
        return StreamedResult(returncode=code, last_line="", output_tail=[])

    monkeypatch.setattr(environment_manager, "stream_with_tail", fake_stream)
    mgr = EnvironmentManager(envs_dir=tmp_path / "envs", micromamba_path=micromamba)
    mgr._create_env("probe", tmp_path / "envs" / "env-probe", yaml_path, job_id=job_id)
    return runs


@pytest.mark.parametrize("crash", [WINDOWS_ACCESS_VIOLATION, -signal.SIGSEGV])
def test_a_crash_is_retried_once_without_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash: int
) -> None:
    runs = build(tmp_path, monkeypatch, [crash, 0])

    assert len(runs) == 2
    assert "MAMBA_USE_SHARDED_REPODATA" not in runs[0]
    assert runs[1]["MAMBA_USE_SHARDED_REPODATA"] == "false"
    assert (tmp_path / "envs" / "env-probe").is_dir()


def test_a_second_crash_is_reported_not_retried_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match=str(WINDOWS_ACCESS_VIOLATION)):
        build(tmp_path, monkeypatch, [WINDOWS_ACCESS_VIOLATION] * 2)


def test_an_ordinary_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pip error or a solver conflict fails the same way the second time;
    retrying would only double the wait before the user sees it."""
    with pytest.raises(RuntimeError, match="exit 1"):
        build(tmp_path, monkeypatch, [1, 0])


def test_a_cancel_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(environment_manager, "is_cancel_requested", lambda job_id: True)
    with pytest.raises(JobCancelledError):
        build(tmp_path, monkeypatch, [WINDOWS_ACCESS_VIOLATION, 0], job_id="job")


def test_no_retry_into_a_half_built_env_that_cannot_be_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building into a leftover half-env is what made retries stall; when
    the crashed run's temp env cannot be removed, report the crash."""
    monkeypatch.setattr(
        EnvironmentManager, "_discard_temp_env", lambda self, path, why: False
    )
    with pytest.raises(RuntimeError, match=str(WINDOWS_ACCESS_VIOLATION)):
        build(tmp_path, monkeypatch, [WINDOWS_ACCESS_VIOLATION, 0])
