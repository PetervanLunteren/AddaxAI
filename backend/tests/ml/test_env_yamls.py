"""Content checks for the bundled environment.yml files.

The yml files under app/ml/envs/ are executed by micromamba on user
machines, so a syntax error or a regressed dependency line only
surfaces at install time in the field. These tests catch that in CI.
"""

from pathlib import Path

import pytest
import yaml

ENVS_DIR = Path(__file__).resolve().parents[2] / "app" / "ml" / "envs"

ALL_YAMLS = sorted(ENVS_DIR.glob("*/*/environment.yml"))

ADDAXAI_BASE_PLATFORMS = ("darwin", "linux", "windows")


def _pip_deps(yaml_path: Path) -> list[str]:
    """Return the pip dependency strings from an environment.yml."""
    data = yaml.safe_load(yaml_path.read_text())
    for dep in data["dependencies"]:
        if isinstance(dep, dict) and "pip" in dep:
            return dep["pip"]
    raise AssertionError(f"no pip section in {yaml_path}")


def test_all_environment_yamls_parse() -> None:
    """Every bundled yml is valid YAML with a name and dependencies."""
    assert ALL_YAMLS, f"no environment.yml files found under {ENVS_DIR}"
    for yaml_path in ALL_YAMLS:
        data = yaml.safe_load(yaml_path.read_text())
        assert data["name"].startswith("env-"), yaml_path
        assert data["dependencies"], yaml_path


@pytest.mark.parametrize("platform_dir", ADDAXAI_BASE_PLATFORMS)
def test_addaxai_base_pins_ultralytics_yolov5_wheel(platform_dir: str) -> None:
    """
    ultralytics-yolov5 must be installed from our prebuilt wheel, not
    the PyPI sdist. The sdist's setup.py downloads a README from GitHub
    at build time, which crashes on machines where Python cannot load
    the Windows certificate store (beta report 2026-06-10). A direct
    wheel URL skips setup.py; the #sha256= fragment makes pip verify
    the artifact.
    """
    yaml_path = ENVS_DIR / "addaxai-base" / platform_dir / "environment.yml"
    pip_deps = _pip_deps(yaml_path)

    wheel_refs = [
        d for d in pip_deps if d.startswith("ultralytics-yolov5 @ https://")
    ]
    assert len(wheel_refs) == 1, (
        f"{yaml_path} must reference ultralytics-yolov5 as a direct wheel "
        f"URL exactly once, found: {wheel_refs}"
    )

    url, _, fragment = wheel_refs[0].partition("#")
    assert url.endswith(".whl"), wheel_refs[0]
    assert fragment.startswith("sha256="), (
        f"{yaml_path}: wheel reference must carry a #sha256= fragment"
    )

    assert any(d.startswith("megadetector==") for d in pip_deps), (
        f"{yaml_path}: megadetector pin missing"
    )


def test_docs_torch_wheel_link_matches_windows_pin() -> None:
    """
    The "install in mainland China" docs page links the Windows torch
    wheel by exact file name, so users behind the firewall can download
    it directly. That name embeds the torch version, CUDA tag and
    Python version pinned here, so a bump that skips the docs would
    leave the page pointing at the wrong wheel, silently.
    """
    yaml_path = ENVS_DIR / "addaxai-base" / "windows" / "environment.yml"
    data = yaml.safe_load(yaml_path.read_text())

    python_pins = [
        d
        for d in data["dependencies"]
        if isinstance(d, str) and d.startswith("python=")
    ]
    assert len(python_pins) == 1, python_pins
    major, minor = python_pins[0].removeprefix("python=").split(".")[:2]

    torch_pins = [d for d in _pip_deps(yaml_path) if d.startswith("torch==")]
    assert len(torch_pins) == 1, torch_pins
    version = torch_pins[0].removeprefix("torch==")

    wheel_name = (
        f"torch-{version.replace('+', '%2B')}"
        f"-cp{major}{minor}-cp{major}{minor}-win_amd64.whl"
    )
    docs_page = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "docs"
        / "help"
        / "install-in-china.mdx"
    )
    assert wheel_name in docs_page.read_text(), (
        f"{docs_page.name} must link the pinned Windows torch wheel "
        f"{wheel_name}; update the download link after a torch or "
        f"python bump"
    )
