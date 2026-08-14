"""
Tests for the bundled-wheel substitution applied to the env YAML copy
before micromamba runs.

pip fetches a direct-URL requirement literally, so no index setting can
redirect it. Shipping the wheel is the only thing that makes the env
build work on a network that blocks the host, which is why these tests
also guard that the file we ship is the file the YAMLs ask for.
"""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

from app.ml.environment_manager import (
    BUNDLED_WHEELS_DIR,
    substitute_bundled_wheels,
)

ENVS_DIR = Path(__file__).resolve().parents[2] / "app" / "ml" / "envs"

_URL_PREFIX = "https://huggingface.co/Addax-Data-Science/pip-wheels/resolve/main/"

_WHEEL_NAME = "ultralytics_yolov5-0.1.1-py3-none-any.whl"

_YAML = (
    "dependencies:\n"
    "  - pip:\n"
    "    - --extra-index-url https://download.pytorch.org/whl/cu128\n"
    "    - torch==2.8.0+cu128\n"
    f"    - ultralytics-yolov5 @ {_URL_PREFIX}{_WHEEL_NAME}"
    "#sha256=d532e62d\n"
)


@pytest.fixture
def wheels_dir(tmp_path: Path) -> Path:
    """A stand-in wheels directory holding the referenced file."""
    (tmp_path / _WHEEL_NAME).write_bytes(b"not a real wheel")
    return tmp_path


def test_url_is_rewritten_to_the_local_file(wheels_dir: Path) -> None:
    out = substitute_bundled_wheels(_YAML, wheels_dir)

    assert "https://huggingface.co/" not in out
    assert f"ultralytics-yolov5 @ {wheels_dir.as_uri()}/{_WHEEL_NAME}" in out


def test_sha256_fragment_survives(wheels_dir: Path) -> None:
    """pip verifies the local copy against it, so it must not be lost."""
    assert "#sha256=d532e62d" in substitute_bundled_wheels(_YAML, wheels_dir)


def test_other_urls_are_untouched(wheels_dir: Path) -> None:
    """The pytorch index is mirrored via pip config, never rewritten here."""
    out = substitute_bundled_wheels(_YAML, wheels_dir)

    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in out
    assert "torch==2.8.0+cu128" in out


def test_yaml_without_a_wheel_is_unchanged(wheels_dir: Path) -> None:
    text = "dependencies:\n  - python=3.11\n"

    assert substitute_bundled_wheels(text, wheels_dir) == text


def test_missing_wheel_raises_and_names_the_file(tmp_path: Path) -> None:
    """
    A wheel that did not make it into the build is a packaging defect.
    Fail loudly here rather than let pip fail deep inside micromamba
    output, or silently fall back to a host the user cannot reach.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        substitute_bundled_wheels(_YAML, tmp_path)

    assert _WHEEL_NAME in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


def _wheels_referenced_by(yaml_path: Path) -> list[str]:
    """Wheel filenames the YAML pins by direct URL."""
    pattern = re.compile(re.escape(_URL_PREFIX) + r"([^\s#]+)")
    return pattern.findall(yaml_path.read_text(encoding="utf-8"))


def _sha256_in(yaml_path: Path, wheel_name: str) -> str | None:
    """The #sha256= fragment the YAML declares for that wheel."""
    pattern = re.compile(
        re.escape(_URL_PREFIX + wheel_name) + r"#sha256=([a-f0-9]+)"
    )
    match = pattern.search(yaml_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def test_every_pinned_wheel_is_bundled_and_matches_its_hash() -> None:
    """
    The YAMLs stay the single source of truth for which wheel we want.
    This is what stops the shipped file and the pin drifting apart: the
    filename and hash are read out of the YAMLs and checked against what
    is actually in pip-wheels/.
    """
    yaml_paths = sorted(ENVS_DIR.glob("*/*/environment.yml"))
    assert yaml_paths, f"no environment.yml files found under {ENVS_DIR}"

    checked = 0
    for yaml_path in yaml_paths:
        for wheel_name in _wheels_referenced_by(yaml_path):
            wheel = BUNDLED_WHEELS_DIR / wheel_name
            assert wheel.is_file(), (
                f"{yaml_path} pins {wheel_name}, which is not in "
                f"{BUNDLED_WHEELS_DIR}. Add the file or fix the pin."
            )

            expected = _sha256_in(yaml_path, wheel_name)
            assert expected, f"{yaml_path} pins {wheel_name} with no #sha256="

            actual = hashlib.sha256(wheel.read_bytes()).hexdigest()
            assert actual == expected, (
                f"{wheel_name} does not match the sha256 pinned in "
                f"{yaml_path}: expected {expected}, got {actual}."
            )
            checked += 1

    assert checked, "no bundled wheel is pinned by any environment.yml"


def test_bundled_wheels_are_all_referenced() -> None:
    """
    Nothing sits in pip-wheels/ that no environment asks for. A wheel no
    YAML pins is dead weight in every installer on every platform.
    """
    pinned = {
        name
        for yaml_path in ENVS_DIR.glob("*/*/environment.yml")
        for name in _wheels_referenced_by(yaml_path)
    }
    shipped = {p.name for p in BUNDLED_WHEELS_DIR.glob("*.whl")}

    assert shipped == pinned


def test_pinned_wheel_satisfies_the_pip_requirement_name() -> None:
    """
    The requirement is `ultralytics-yolov5 @ <url>`, so the file has to
    be a wheel for that distribution or pip installs the wrong thing.
    """
    for yaml_path in sorted(ENVS_DIR.glob("*/*/environment.yml")):
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        pip_deps = next(
            (
                dep["pip"]
                for dep in data["dependencies"]
                if isinstance(dep, dict) and "pip" in dep
            ),
            [],
        )
        for dep in pip_deps:
            if _URL_PREFIX not in dep:
                continue
            name, _, url = dep.partition(" @ ")
            wheel_name = url.split("#")[0].rsplit("/", 1)[-1]
            assert wheel_name.startswith(name.replace("-", "_")), (
                f"{yaml_path}: requirement {name!r} is pinned to "
                f"{wheel_name!r}, which is a different distribution."
            )
