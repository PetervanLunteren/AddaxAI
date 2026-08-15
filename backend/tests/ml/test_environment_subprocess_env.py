"""Tests for the environment micromamba build inherits.

The build's output pipe is read as UTF-8 (`stream_with_tail`). micromamba
itself is a native binary and emits UTF-8 on every platform, but the pip
it runs is Python and follows the system codepage, so the encoding it
writes has to be forced or the two halves of one stream disagree. That
only shows up on a non-UTF-8 locale, which no CI runner and no developer
machine here has, so it is pinned rather than observed.
"""

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
  - pip
  - pip:
    - tqdm
"""


def capture_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """Run `_create_env` far enough to capture the env it hands micromamba."""
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text(YAML)

    # A file that already exists, so `_ensure_runtime_dirs` does not try
    # to download the real micromamba binary during the test.
    micromamba = tmp_path / "micromamba"
    micromamba.write_text("")

    seen: dict[str, Any] = {}

    def fake_stream(cmd: list[str], **kwargs: Any) -> StreamedResult:
        seen.update(kwargs)
        # Non-zero so `_create_env` stops here instead of going on to
        # validate an environment that was never built.
        return StreamedResult(returncode=1, last_line="stopped", output_tail=[])

    monkeypatch.setattr(environment_manager, "stream_with_tail", fake_stream)

    mgr = EnvironmentManager(envs_dir=tmp_path / "envs", micromamba_path=micromamba)
    with pytest.raises(RuntimeError):
        mgr._create_env("probe", tmp_path / "envs" / "env-probe", yaml_path)

    return seen["env"]


@pytest.fixture
def captured_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    return capture_env(tmp_path, monkeypatch)


def test_nested_pip_is_forced_to_utf8(captured_env: dict[str, str]) -> None:
    """Without this, pip's non-ASCII output arrives as U+FFFD on a
    cp932 / cp936 / cp949 machine, losing the traceback at the exact
    moment the install failed."""
    assert captured_env["PYTHONIOENCODING"] == "utf-8"


def test_pip_verbosity_and_retries_survive(captured_env: dict[str, str]) -> None:
    """The encoding knob sits among the pip/mamba knobs; guard against a
    future edit dropping its neighbours."""
    assert captured_env["PIP_VERBOSE"] == "1"
    assert captured_env["MAMBA_REMOTE_MAX_RETRIES"] == "5"


def test_prefer_binary_is_set(captured_env: dict[str, str]) -> None:
    """Falling back to a source package means compiling on the user's
    machine, which needs a C++ compiler a typical Windows user does not
    have. stringzilla 5.1.2 shipped without a cp311 win_amd64 wheel on
    2026-08-12 and blocked every fresh Windows install until it was
    pinned by hand. Preferring an older wheel covers the whole class,
    for every env and platform, from this one place."""
    assert captured_env["PIP_PREFER_BINARY"] == "1"


def test_revocation_check_stays_on_by_default(
    captured_env: dict[str, str],
) -> None:
    """Nothing weakens TLS unless the user has asked for it.

    The variable must be absent, not "false": its presence at all is what
    an audit of a machine would look for."""
    assert "MAMBA_SSL_NO_REVOKE" not in captured_env


def test_marker_disables_the_revocation_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker file is the only thing that turns the check off.

    The value has to be exactly "true". micromamba parses it as YAML and
    "1" dies with a bad-conversion backtrace (mamba issue #2751), which
    would break every build instead of fixing one."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    (tmp_path / environment_manager.REVOCATION_MARKER_FILENAME).write_text("x")

    env = capture_env(tmp_path, monkeypatch)

    assert env["MAMBA_SSL_NO_REVOKE"] == "true"


def test_user_site_packages_stay_out(captured_env: dict[str, str]) -> None:
    """`clean_python_env` still applies: the build must not pick up the
    user's own site-packages or a globally exported PYTHONPATH."""
    assert captured_env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in captured_env
