"""Tests for the C++ runtime shipped next to micromamba on Windows.

The Windows micromamba is not self-contained: it loads Microsoft's C++
runtime (msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll), and it is
built with toolset 14.44. A machine whose own msvcp140.dll is older than
14.40 makes it die at start-up with 0xC0000005 and no output, inside
std::mutex. Users on 7.11.0 to 7.13.0 hit exactly that at the first setup
step (October 2026). Windows looks in the program's own directory first,
so a copy next to micromamba.exe wins over whatever the machine has.
"""

import hashlib
from pathlib import Path

import pytest

from app.ml import environment_manager
from app.ml.environment_manager import (
    VC_RUNTIME_DIR,
    VC_RUNTIME_DLLS,
    EnvironmentManager,
)

# vc14_runtime 14.44.35208 h818238b_41 from conda-forge, the toolset the
# pinned micromamba was built with. See vc-runtime/README.md.
SHIPPED_SHA256 = {
    "msvcp140.dll": "8f141b4454fa78db34bc1f28c571b4da0e00cd2c43f7ad0e282f313036826aae",
    "vcruntime140.dll": "0205071c36c17f1efbd70178c852cb7d49985c484202752b8704b7ac6b184e60",
    "vcruntime140_1.dll": "963e45edd064545962e216c12d68071ced94dc8e11862a18f07f14eb2690a57c",
}


def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str) -> EnvironmentManager:
    """A manager on `system` whose micromamba already exists (no download)."""
    monkeypatch.setattr(environment_manager.platform, "system", lambda: system)
    micromamba = tmp_path / "bin" / "2.9.0" / "micromamba.exe"
    micromamba.parent.mkdir(parents=True)
    micromamba.touch()
    return EnvironmentManager(envs_dir=tmp_path / "envs", micromamba_path=micromamba)


def test_the_shipped_runtime_is_the_pinned_build() -> None:
    assert set(VC_RUNTIME_DLLS) == set(SHIPPED_SHA256)
    for name, sha in SHIPPED_SHA256.items():
        assert hashlib.sha256((VC_RUNTIME_DIR / name).read_bytes()).hexdigest() == sha, name


def test_windows_gets_the_runtime_next_to_micromamba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Also covers installs from 7.13.0, which have micromamba but no runtime."""
    mgr = _manager(tmp_path, monkeypatch, "Windows")

    for name in VC_RUNTIME_DLLS:
        installed = mgr.micromamba_path.parent / name
        assert installed.read_bytes() == (VC_RUNTIME_DIR / name).read_bytes()
    assert not list(mgr.micromamba_path.parent.glob("*.copy"))


def test_a_missing_runtime_file_is_put_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checked before every build, like the binary itself, so antivirus
    quarantining one DLL heals on the next attempt."""
    mgr = _manager(tmp_path, monkeypatch, "Windows")
    (mgr.micromamba_path.parent / "msvcp140.dll").unlink()

    mgr._ensure_runtime_dirs()

    assert (mgr.micromamba_path.parent / "msvcp140.dll").is_file()


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
def test_other_platforms_get_no_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    mgr = _manager(tmp_path, monkeypatch, system)

    assert sorted(p.name for p in mgr.micromamba_path.parent.iterdir()) == ["micromamba.exe"]
