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


@pytest.fixture
def captured_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
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


def test_user_site_packages_stay_out(captured_env: dict[str, str]) -> None:
    """`clean_python_env` still applies: the build must not pick up the
    user's own site-packages or a globally exported PYTHONPATH."""
    assert captured_env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in captured_env
