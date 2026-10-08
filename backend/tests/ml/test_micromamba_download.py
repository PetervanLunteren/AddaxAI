"""Tests for how the app obtains its micromamba binary.

Three things are pinned here, each the cure for something that went
wrong in the field:

- One download, however many managers ask at once. Opening the setup
  screen constructs three managers within a second; each downloaded
  micromamba to the same path, and the two late ones failed with a
  500 when they could not overwrite the copy that was already running.
- An exact version, verified by checksum. `latest` let an upstream
  release reach users untested, and nothing checked the bytes.
- The version is in the filename, so a bump reaches existing installs
  too: they download the new build once instead of keeping the old.
"""

import hashlib
import io
import os
import tarfile
import threading
import time
from pathlib import Path

import pytest

from app.ml import environment_manager
from app.ml.environment_manager import (
    MICROMAMBA_FILENAME,
    MICROMAMBA_VERSION,
    EnvironmentManager,
    _micromamba_subdir,
)


def _archive_with_binary(content: bytes) -> bytes:
    """A micromamba-shaped .tar.bz2 holding `content` as the binary."""
    member = (
        "Library/bin/micromamba.exe"
        if _micromamba_subdir().startswith("win")
        else "bin/micromamba"
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _manager_without_binary(tmp_path: Path) -> EnvironmentManager:
    """A manager whose binary is missing, built without a real download."""
    binary = tmp_path / "bin" / MICROMAMBA_FILENAME
    binary.parent.mkdir(parents=True)
    binary.touch()
    mgr = EnvironmentManager(envs_dir=tmp_path / "envs", micromamba_path=binary)
    binary.unlink()
    return mgr


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    """Answer every download with `payload`; return the URLs requested."""
    requested: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        requested.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(environment_manager.urllib.request, "urlopen", fake_urlopen)
    return requested


def test_concurrent_managers_download_micromamba_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "bin" / MICROMAMBA_FILENAME
    calls = 0

    def slow_download(self: EnvironmentManager) -> None:
        nonlocal calls
        calls += 1
        time.sleep(0.2)  # wide enough for every thread to see it missing
        self.micromamba_path.write_bytes(b"binary")

    monkeypatch.setattr(EnvironmentManager, "_download_micromamba", slow_download)

    def construct() -> None:
        EnvironmentManager(envs_dir=tmp_path / "envs", micromamba_path=binary)

    threads = [threading.Thread(target=construct) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == 1
    assert binary.read_bytes() == b"binary"


def test_a_verified_archive_installs_the_pinned_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive_with_binary(b"the real binary")
    monkeypatch.setitem(
        environment_manager._MICROMAMBA_SHA256,
        _micromamba_subdir(),
        hashlib.sha256(archive).hexdigest(),
    )
    requested = _serve(monkeypatch, archive)
    mgr = _manager_without_binary(tmp_path)

    mgr._download_micromamba()

    assert mgr.micromamba_path.read_bytes() == b"the real binary"
    if os.name == "posix":
        assert os.access(mgr.micromamba_path, os.X_OK)
    assert requested == [
        f"https://micro.mamba.pm/api/micromamba/{_micromamba_subdir()}/{MICROMAMBA_VERSION}"
    ]
    assert list(mgr.micromamba_path.parent.iterdir()) == [mgr.micromamba_path]


def test_a_download_that_fails_its_checksum_installs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, _archive_with_binary(b"tampered or truncated"))
    mgr = _manager_without_binary(tmp_path)

    with pytest.raises(RuntimeError, match="checksum"):
        mgr._download_micromamba()

    # Nothing left that a later run could mistake for an installed binary.
    assert list(mgr.micromamba_path.parent.iterdir()) == []


@pytest.mark.parametrize(
    ("system", "machine", "subdir"),
    [
        ("Windows", "AMD64", "win-64"),
        ("Darwin", "arm64", "osx-arm64"),
        ("Darwin", "x86_64", "osx-64"),
        ("Linux", "x86_64", "linux-64"),
        ("Linux", "aarch64", "linux-aarch64"),
    ],
)
def test_every_supported_platform_has_a_pinned_checksum(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, subdir: str
) -> None:
    monkeypatch.setattr(environment_manager.platform, "system", lambda: system)
    monkeypatch.setattr(environment_manager.platform, "machine", lambda: machine)
    assert _micromamba_subdir() == subdir
    assert len(environment_manager._MICROMAMBA_SHA256[subdir]) == 64
