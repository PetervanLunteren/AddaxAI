"""ffmpeg is resolved from the analysis environment first, PATH second."""

from types import SimpleNamespace

import pytest

from app.utils import ffmpeg_bin


def test_the_envs_own_binary_wins(tmp_path, monkeypatch):
    env_bin = tmp_path / "envs" / "env-marine" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "ffmpeg").write_bytes(b"")
    monkeypatch.setattr(ffmpeg_bin, "get_settings", lambda: SimpleNamespace(user_data_dir=tmp_path))
    monkeypatch.setattr(ffmpeg_bin.os, "name", "posix")

    assert ffmpeg_bin.resolve_ffmpeg("marine") == str(env_bin / "ffmpeg")


def test_path_is_the_fallback_and_nothing_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ffmpeg_bin, "get_settings", lambda: SimpleNamespace(user_data_dir=tmp_path))
    monkeypatch.setattr(ffmpeg_bin.shutil, "which", lambda name: "/opt/ffmpeg")
    assert ffmpeg_bin.resolve_ffmpeg("marine") == "/opt/ffmpeg"

    monkeypatch.setattr(ffmpeg_bin.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="env-marine"):
        ffmpeg_bin.resolve_ffmpeg("marine")
