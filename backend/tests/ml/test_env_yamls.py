"""Content checks for the bundled environment.yml files.

The yml files under app/ml/envs/ are executed by micromamba on user
machines, so a syntax error or a regressed dependency line only
surfaces at install time in the field. These tests catch that in CI.
"""

import re
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


# The envs a detector runs in: each carries megadetector and the app's own
# tracker, since the tracking script runs inside the detector's env.
DETECTOR_ENVS = ("addaxai-base", "marine", "ultralytics")


@pytest.mark.parametrize("platform_dir", ADDAXAI_BASE_PLATFORMS)
@pytest.mark.parametrize("env", DETECTOR_ENVS)
def test_detector_envs_pin_the_yolov5_wheel_and_megadetector(env: str, platform_dir: str) -> None:
    """
    ultralytics-yolov5 must be installed from our prebuilt wheel, not
    the PyPI sdist. The sdist's setup.py downloads a README from GitHub
    at build time, which crashes on machines where Python cannot load
    the Windows certificate store (beta report 2026-06-10). A direct
    wheel URL skips setup.py; the #sha256= fragment makes pip verify
    the artifact.
    """
    yaml_path = ENVS_DIR / env / platform_dir / "environment.yml"
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


@pytest.mark.parametrize("platform_dir", ADDAXAI_BASE_PLATFORMS)
@pytest.mark.parametrize("env", DETECTOR_ENVS)
def test_detector_envs_carry_the_trackers_solvers(env: str, platform_dir: str) -> None:
    """The tracking script's own BoT-SORT imports lap and scipy in every
    detector env. Neither is a declared dependency of anything the app
    installs there on purpose (scipy only arrives through scikit-learn),
    so each must be pinned, or a video analysis dies on import."""
    pip_deps = _pip_deps(ENVS_DIR / env / platform_dir / "environment.yml")
    assert "lap==0.5.12" in pip_deps
    assert "scipy==1.17.1" in pip_deps


def _yaml_id(yaml_path: Path) -> str:
    return f"{yaml_path.parent.parent.name}-{yaml_path.parent.name}"


@pytest.mark.parametrize("yaml_path", ALL_YAMLS, ids=_yaml_id)
def test_ultralytics_is_installed_only_in_its_own_env(yaml_path: Path) -> None:
    """ultralytics is AGPL-3.0. The app asks the user to accept that before
    it builds env-ultralytics, so the package must be in that env and in
    no other: one stray line and every user gets it without being asked.
    The yolov5 snapshot wheel is a different package and does not count."""
    has_it = any(
        re.match(r"ultralytics\s*(==|>=|<=|~=|$)", dep) for dep in _pip_deps(yaml_path)
    )
    assert has_it == (yaml_path.parent.parent.name == "ultralytics"), yaml_path


def test_drift_is_checked_for_every_env_the_build_ships() -> None:
    """`shipped_env_names` is what the "your environment is out of date"
    notice iterates, and it must cover every bundled recipe.

    This used to be a hand-written tuple in `catalog_updater`, which is a
    silent failure waiting to happen: nothing else reads the list, so an
    env missing from it is simply never checked and no error appears
    anywhere. Deriving it from the recipes is the fix; this pins that the
    two agree, including that the derivation sees every platform's env.
    """
    from app.ml.environment_manager import ENV_RECIPES_DIR, shipped_env_names

    assert ENV_RECIPES_DIR == ENVS_DIR
    on_disk = sorted({p.parent.parent.name for p in ALL_YAMLS})
    assert shipped_env_names() == on_disk
    # Detector envs are exactly the case the old hand-written list would
    # have missed.
    assert {"marine", "ultralytics"} <= set(shipped_env_names())


@pytest.mark.parametrize("platform_dir", ADDAXAI_BASE_PLATFORMS)
def test_pytorch_env_pins_pandas(platform_dir: str) -> None:
    """Four classifiers' inference.py import pandas. It used to come in
    through ultralytics; since that moved to its own env, nothing else in
    env-pytorch brings it, and those models would fail on import."""
    pip_deps = _pip_deps(ENVS_DIR / "pytorch" / platform_dir / "environment.yml")
    assert any(d.startswith("pandas==") for d in pip_deps)
