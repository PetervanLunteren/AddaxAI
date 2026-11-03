"""Tests for the non-ASCII data folder failure on Windows.

micromamba cannot open its spec file when the path holds a character
outside ASCII: it converts its command line to UTF-8 and checks that
the file exists through the wide Windows API, then hands the UTF-8
string to yaml-cpp's LoadFile, which opens it through the narrow C
runtime in the ANSI code page. A Windows user folder with an ñ killed
the first setup step that way (beta report 2026-10-23). The failure
cannot be produced on a developer machine, so the detection is driven
with the exact tail that run printed, and the platform is patched.
"""

from pathlib import Path
from typing import Any

import pytest

from app.ml import environment_manager
from app.ml.environment_manager import (
    EnvironmentManager,
    NonAsciiPathError,
    is_non_ascii_path_failure,
)
from app.utils.subprocess_runner import StreamedResult

YAML = """name: env-probe
channels:
  - conda-forge
dependencies:
  - python=3.11
"""

# Verbatim from the screenshot of the failing Windows install.
SPEC_OPEN_FAILED_TAIL = [
    "trace    libmamba Compute configurable 'envs_dirs'",
    "trace    libmamba Compute configurable 'file_specs'",
    "error    libmamba YAML error in spec file "
    "'C:\\Users\\LuisFelipeAcuñaAlvar\\AddaxAI\\envs\\.env-addaxai-base.environment.yml'",
    "critical libmamba bad file: "
    "C:\\Users\\LuisFelipeAcuñaAlvar\\AddaxAI\\envs\\.env-addaxai-base.environment.yml",
    "info     libmamba ****************** Backtrace End ********************",
]

PIP_FAILED_TAIL = [
    "ERROR: Could not install packages due to an OSError",
    "critical libmamba pip failed to install packages",
]

ACCENTED_SPEC = Path("C:/Users/LuisFelipeAcuñaAlvar/AddaxAI/envs/.env.environment.yml")
PLAIN_SPEC = Path("C:/Users/Luis/AddaxAI/envs/.env.environment.yml")


def test_an_unopenable_spec_on_an_accented_path_is_recognised() -> None:
    assert is_non_ascii_path_failure(SPEC_OPEN_FAILED_TAIL, ACCENTED_SPEC)


def test_the_same_output_on_a_plain_path_is_not() -> None:
    assert not is_non_ascii_path_failure(SPEC_OPEN_FAILED_TAIL, PLAIN_SPEC)


def test_an_ordinary_failure_on_an_accented_path_is_not() -> None:
    assert not is_non_ascii_path_failure(PIP_FAILED_TAIL, ACCENTED_SPEC)


def failing_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> tuple[EnvironmentManager, Path, Path]:
    """A manager whose data folder carries an ñ, as on the reported
    machine, and whose micromamba exits 1 with the spec-open tail."""
    data_dir = tmp_path / "Acuña"
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text(YAML, encoding="utf-8")
    micromamba = tmp_path / "micromamba"
    micromamba.write_text("")

    def fake_stream(cmd: list[str], **kwargs: Any) -> StreamedResult:
        return StreamedResult(
            returncode=1,
            last_line=SPEC_OPEN_FAILED_TAIL[-1],
            output_tail=list(SPEC_OPEN_FAILED_TAIL),
        )

    monkeypatch.setattr(environment_manager, "stream_with_tail", fake_stream)
    monkeypatch.setattr(environment_manager.platform, "system", lambda: system)
    mgr = EnvironmentManager(envs_dir=data_dir / "envs", micromamba_path=micromamba)
    return mgr, data_dir / "envs" / "env-probe", yaml_path


def test_windows_build_on_an_accented_path_raises_the_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr, env_path, yaml_path = failing_build(tmp_path, monkeypatch, "Windows")

    with pytest.raises(NonAsciiPathError) as exc:
        mgr._create_env("probe", env_path, yaml_path)

    message = str(exc.value)
    assert str(tmp_path / "Acuña") in message
    assert "(ñ)" in message
    assert "ADDAXAI_USER_DATA_DIR" in message
    assert "bad file" not in message


def test_the_same_failure_elsewhere_stays_the_generic_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr, env_path, yaml_path = failing_build(tmp_path, monkeypatch, "Darwin")

    with pytest.raises(RuntimeError) as exc:
        mgr._create_env("probe", env_path, yaml_path)

    assert not isinstance(exc.value, NonAsciiPathError)
    assert "micromamba create failed (exit 1)" in str(exc.value)
