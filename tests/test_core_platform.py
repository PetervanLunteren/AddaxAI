"""Tests for addaxai.core.platform — OS detection and interpreter lookup."""

import os
import platform
import pytest

from addaxai.core.platform import get_python_interpreter


def test_get_python_interpreter_windows(tmp_path):
    """On Windows, interpreter should be in envs/env-{name}/python.exe."""
    result = get_python_interpreter(str(tmp_path), "base", system="Windows")
    expected = os.path.join(str(tmp_path), "envs", "env-base", "python.exe")
    assert result == expected


def test_get_python_interpreter_unix(tmp_path):
    """On non-Windows, interpreter should be in envs/env-{name}/bin/python."""
    result = get_python_interpreter(str(tmp_path), "base", system="Linux")
    expected = os.path.join(str(tmp_path), "envs", "env-base", "bin", "python")
    assert result == expected


def test_get_python_interpreter_macos(tmp_path):
    """On macOS, interpreter should be in envs/env-{name}/bin/python."""
    result = get_python_interpreter(str(tmp_path), "base", system="Darwin")
    expected = os.path.join(str(tmp_path), "envs", "env-base", "bin", "python")
    assert result == expected


def test_get_python_interpreter_default_uses_current_system(tmp_path):
    """When system is not specified, it should use the current platform."""
    result = get_python_interpreter(str(tmp_path), "tensorflow")
    if platform.system() == "Windows":
        assert result.endswith("python.exe")
    else:
        assert result.endswith(os.path.join("bin", "python"))
