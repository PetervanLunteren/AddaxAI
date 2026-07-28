"""Tests for YAML-hash drift detection in app.ml.environment_manager."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.ml.environment_manager import (
    ENV_YAML_SHA_FILENAME,
    EnvironmentManager,
    hash_yaml_file,
)


@pytest.fixture
def envs_dir(tmp_path: Path) -> Path:
    return tmp_path / "envs"


@pytest.fixture
def env_manager(envs_dir: Path) -> EnvironmentManager:
    """
    EnvironmentManager pointed at a temp envs dir. micromamba isn't
    actually used in these tests because we're poking at the hash
    sentinel directly, so we patch out _ensure_runtime_dirs to avoid
    the real micromamba download.
    """
    with patch.object(
        EnvironmentManager, "_ensure_runtime_dirs", lambda self: None
    ):
        return EnvironmentManager(envs_dir=envs_dir)


def test_hash_yaml_file_is_deterministic(tmp_path: Path) -> None:
    """Same bytes → same hex digest."""
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text("name: test\ndependencies:\n  - python=3.11\n")
    h1 = hash_yaml_file(yaml_path)
    h2 = hash_yaml_file(yaml_path)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest length


def test_hash_yaml_file_changes_on_any_byte_edit(tmp_path: Path) -> None:
    """Even a comment edit changes the hash. Wanted behaviour."""
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text("name: test\n")
    before = hash_yaml_file(yaml_path)
    yaml_path.write_text("name: test\n# trailing comment\n")
    after = hash_yaml_file(yaml_path)
    assert before != after


def test_check_yaml_drift_returns_none_for_missing_env(
    env_manager: EnvironmentManager,
) -> None:
    """No env on disk → unknown but valid → skip drift check."""
    assert env_manager.check_yaml_drift("nonexistent") is None


def test_check_yaml_drift_returns_none_for_legacy_install(
    env_manager: EnvironmentManager, envs_dir: Path
) -> None:
    """
    Env exists but has no sentinel file (legacy install predating drift
    detection). Per the design call, treat as unknown but valid rather
    than as drift.
    """
    env_path = envs_dir / "env-test"
    env_path.mkdir(parents=True)
    assert env_manager.check_yaml_drift("test") is None


def test_check_yaml_drift_returns_false_when_hash_matches(
    env_manager: EnvironmentManager, envs_dir: Path, tmp_path: Path
) -> None:
    """Sentinel matches current bundled YAML → in sync."""
    env_path = envs_dir / "env-test"
    env_path.mkdir(parents=True)
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text("name: test\n")

    sentinel = env_path / ENV_YAML_SHA_FILENAME
    sentinel.write_text(hash_yaml_file(yaml_path))

    with patch.object(
        EnvironmentManager,
        "get_env_yaml_path",
        return_value=yaml_path,
    ):
        result = env_manager.check_yaml_drift("test")
    assert result is False


def test_check_yaml_drift_returns_true_when_yaml_moved(
    env_manager: EnvironmentManager, envs_dir: Path, tmp_path: Path
) -> None:
    """Sentinel was written for an older YAML; YAML has since moved."""
    env_path = envs_dir / "env-test"
    env_path.mkdir(parents=True)
    yaml_path = tmp_path / "environment.yml"

    yaml_path.write_text("name: test\n# v1\n")
    sentinel = env_path / ENV_YAML_SHA_FILENAME
    sentinel.write_text(hash_yaml_file(yaml_path))

    # YAML moves on after the env was built.
    yaml_path.write_text("name: test\n# v2 with extra dep\n")

    with patch.object(
        EnvironmentManager,
        "get_env_yaml_path",
        return_value=yaml_path,
    ):
        result = env_manager.check_yaml_drift("test")
    assert result is True


def test_check_yaml_drift_returns_none_when_yaml_unreadable(
    env_manager: EnvironmentManager, envs_dir: Path
) -> None:
    """Bundled YAML missing → can't compute, treat as unknown."""
    env_path = envs_dir / "env-test"
    env_path.mkdir(parents=True)
    sentinel = env_path / ENV_YAML_SHA_FILENAME
    sentinel.write_text("a" * 64)

    with patch.object(
        EnvironmentManager,
        "get_env_yaml_path",
        side_effect=FileNotFoundError("no such file"),
    ):
        result = env_manager.check_yaml_drift("test")
    assert result is None
