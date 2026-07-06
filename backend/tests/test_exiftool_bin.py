"""Tests for exiftool binary resolution."""

import os
import stat
from types import SimpleNamespace

import pytest

from app.utils import exiftool_bin


def _fake_settings(tmp_path):
    return SimpleNamespace(user_data_dir=tmp_path)


@pytest.fixture
def skip_preflight(monkeypatch):
    """Resolution tests use fake binaries; skip the -ver pre-flight.
    Dedicated tests below exercise it explicitly."""
    monkeypatch.setattr(exiftool_bin, "_verify_spawnable", lambda p: p)


def test_resolves_env_binary_first(tmp_path, monkeypatch, skip_preflight):
    """The env-addaxai-base binary wins over PATH, and the env's bin is
    prepended to PATH so the script's `#!/usr/bin/env perl` shebang
    resolves the env's perl instead of the system one."""
    env_bin = tmp_path / "envs" / "env-addaxai-base" / "bin"
    env_bin.mkdir(parents=True)
    binary = env_bin / "exiftool"
    binary.write_text("#!/usr/bin/env perl\n")

    monkeypatch.setattr(
        exiftool_bin, "get_settings", lambda: _fake_settings(tmp_path)
    )
    monkeypatch.setattr(
        exiftool_bin.shutil, "which", lambda _: "/usr/bin/exiftool"
    )
    monkeypatch.setenv("PATH", "/usr/bin")

    assert exiftool_bin.resolve_exiftool() == str(binary)
    path_parts = exiftool_bin.os.environ["PATH"].split(
        exiftool_bin.os.pathsep
    )
    assert path_parts[0] == str(env_bin)

    # Second resolve must not duplicate the PATH entry.
    exiftool_bin.resolve_exiftool()
    path_parts = exiftool_bin.os.environ["PATH"].split(
        exiftool_bin.os.pathsep
    )
    assert path_parts.count(str(env_bin)) == 1


def test_falls_back_to_path(tmp_path, monkeypatch, skip_preflight):
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


def test_windows_uses_bat_and_extends_path(tmp_path, monkeypatch, skip_preflight):
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


def _make_script(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(os.name == "nt", reason="posix shell scripts")
def test_verify_spawnable_accepts_working_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(exiftool_bin, "_verified", set())
    good = _make_script(tmp_path / "exiftool", "#!/bin/sh\necho 13.59\n")

    assert exiftool_bin._verify_spawnable(str(good)) == str(good)
    assert str(good) in exiftool_bin._verified


@pytest.mark.skipif(os.name == "nt", reason="posix shell scripts")
def test_verify_spawnable_raises_with_stderr(tmp_path, monkeypatch):
    """A binary that dies at startup fails loudly with its stderr,
    instead of PyExifTool later hanging on the dead child."""
    monkeypatch.setattr(exiftool_bin, "_verified", set())
    bad = _make_script(
        tmp_path / "exiftool",
        "#!/bin/sh\necho 'Cannot locate Image/ExifTool.pm' >&2\nexit 2\n",
    )

    with pytest.raises(RuntimeError, match="Image/ExifTool.pm"):
        exiftool_bin._verify_spawnable(str(bad))
