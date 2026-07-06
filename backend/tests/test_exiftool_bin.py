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
