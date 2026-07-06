"""Tests for exiftool binary resolution."""

from types import SimpleNamespace

import pytest

from app.utils import exiftool_bin


def _fake_settings(tmp_path):
    return SimpleNamespace(user_data_dir=tmp_path)


def test_resolves_env_binary_first(tmp_path, monkeypatch):
    """The env-addaxai-base binary wins over PATH."""
    env_bin = tmp_path / "envs" / "env-addaxai-base" / "bin"
    env_bin.mkdir(parents=True)
    binary = env_bin / "exiftool"
    binary.write_text("#!/bin/sh\n")

    monkeypatch.setattr(
        exiftool_bin, "get_settings", lambda: _fake_settings(tmp_path)
    )
    monkeypatch.setattr(
        exiftool_bin.shutil, "which", lambda _: "/usr/bin/exiftool"
    )

    assert exiftool_bin.resolve_exiftool() == str(binary)


def test_falls_back_to_path(tmp_path, monkeypatch):
    """Without an env binary, PATH resolution applies (dev machines, CI)."""
    monkeypatch.setattr(
        exiftool_bin, "get_settings", lambda: _fake_settings(tmp_path)
    )
    monkeypatch.setattr(
        exiftool_bin.shutil, "which", lambda _: "/usr/bin/exiftool"
    )

    assert exiftool_bin.resolve_exiftool() == "/usr/bin/exiftool"


def test_raises_when_missing_everywhere(tmp_path, monkeypatch):
    """Missing binary raises with an actionable message, never silent."""
    monkeypatch.setattr(
        exiftool_bin, "get_settings", lambda: _fake_settings(tmp_path)
    )
    monkeypatch.setattr(exiftool_bin.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="exiftool not found"):
        exiftool_bin.resolve_exiftool()


def test_windows_uses_bat_and_extends_path(tmp_path, monkeypatch):
    """On Windows the .bat wrapper wins (the extensionless perl script
    also exists but is not executable there: WinError 193), and the
    env's binary dirs land on PATH so the wrapper can find perl.exe."""
    env_dir = tmp_path / "envs" / "env-addaxai-base"
    env_bin = env_dir / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "exiftool").write_text("#!perl\n")
    bat = env_bin / "exiftool.bat"
    bat.write_text("@perl -x -S %0 %*\n")
    lib_bin = env_dir / "Library" / "bin"
    lib_bin.mkdir(parents=True)

    monkeypatch.setattr(
        exiftool_bin, "get_settings", lambda: _fake_settings(tmp_path)
    )
    monkeypatch.setattr(exiftool_bin.os, "name", "nt")
    monkeypatch.setenv("PATH", "/usr/bin")

    assert exiftool_bin.resolve_exiftool() == str(bat)
    path_parts = exiftool_bin.os.environ["PATH"].split(
        exiftool_bin.os.pathsep
    )
    assert str(lib_bin) in path_parts
    assert str(env_bin) in path_parts

    # Second resolve must not duplicate the PATH entries.
    exiftool_bin.resolve_exiftool()
    path_parts = exiftool_bin.os.environ["PATH"].split(
        exiftool_bin.os.pathsep
    )
    assert path_parts.count(str(lib_bin)) == 1
