"""Tests for addaxai.core.paths — path resolution functions."""

import os
import pytest

from addaxai.core.paths import get_base_path, get_cls_dir, get_det_dir, get_env_dir, get_version


def test_get_base_path_returns_parent_of_parent(tmp_path):
    """get_base_path should return the grandparent of the given script file."""
    script_dir = tmp_path / "AddaxAI"
    script_dir.mkdir()
    script_file = script_dir / "AddaxAI_GUI.py"
    script_file.touch()

    result = get_base_path(str(script_file))
    assert os.path.normpath(result) == os.path.normpath(str(tmp_path))


def test_get_cls_dir(tmp_path):
    """get_cls_dir should return base/models/cls."""
    result = get_cls_dir(str(tmp_path))
    assert result == os.path.join(str(tmp_path), "models", "cls")


def test_get_det_dir(tmp_path):
    """get_det_dir should return base/models/det."""
    result = get_det_dir(str(tmp_path))
    assert result == os.path.join(str(tmp_path), "models", "det")


def test_get_env_dir(tmp_path):
    """get_env_dir should return base/envs."""
    result = get_env_dir(str(tmp_path))
    assert result == os.path.join(str(tmp_path), "envs")


def test_get_version(tmp_path):
    """get_version should read version.txt and return stripped contents."""
    addaxai_dir = tmp_path / "AddaxAI"
    addaxai_dir.mkdir()
    (addaxai_dir / "version.txt").write_text("  3.5.1  \n")

    result = get_version(str(tmp_path))
    assert result == "3.5.1"
