"""
Locate the exiftool binary.

The installers do not bundle exiftool and fresh machines do not have it
on PATH, so the binary ships inside the env-addaxai-base micromamba
environment (conda-forge package, available for all three platforms).
Resolution order:

1. The env-addaxai-base environment (production path)
2. PATH (dev machines and CI, which install exiftool system-wide)

Raises RuntimeError when neither is present so callers fail loudly with
an actionable message instead of PyExifTool's generic "not found".

The conda exiftool is a perl script and its perl resolution needs help
on every platform, because nothing ever activates the conda env for our
backend process:

- POSIX: the script's shebang is ``#!/usr/bin/env perl`` (conda-forge
  does not rewrite it), so a plain spawn resolves the *system* perl,
  which cannot see the env's ExifTool modules ("Can't locate
  Image/ExifTool.pm in @INC", Linux beta report 2026-07-06). The env's
  own perl 5.32 lives in ``bin/`` next to the script.
- Windows: the win-64 package ships ``bin/exiftool`` (the perl script,
  not directly executable: spawning it raises WinError 193) and
  ``bin/exiftool.bat``, a pl2bat wrapper that invokes plain ``perl``.
  ``perl.exe`` lives in the env's ``Library/bin``.

Both cases have the same fix: prepend the env's binary dirs to this
process's PATH so the right perl wins; children spawned by PyExifTool
inherit it (PyExifTool offers no per-child env parameter). The prepend
means the env's ``bin`` shadows system binaries for this process; that
is acceptable because every other subprocess we spawn uses absolute
paths.
"""

import os
import shutil
import subprocess

from app.core.config import get_settings

# Paths that already passed the -ver pre-flight in this process.
_verified: set[str] = set()


def resolve_exiftool() -> str:
    """Return the absolute path to the exiftool binary."""
    env_dir = get_settings().user_data_dir / "envs" / "env-addaxai-base"

    if os.name == "nt":
        candidate = env_dir / "bin" / "exiftool.bat"
        if candidate.is_file():
            _ensure_env_on_path(
                str(env_dir / "Library" / "bin"),
                str(env_dir / "bin"),
            )
            return _verify_spawnable(str(candidate))
    else:
        candidate = env_dir / "bin" / "exiftool"
        if candidate.is_file():
            _ensure_env_on_path(str(env_dir / "bin"))
            return _verify_spawnable(str(candidate))

    on_path = shutil.which("exiftool")
    if on_path is not None:
        return _verify_spawnable(on_path)

    raise RuntimeError(
        "exiftool not found. Expected it inside the analysis environment "
        f"({env_dir}) or on PATH. Re-run the initial setup to rebuild the "
        "analysis environment."
    )


def _verify_spawnable(path: str) -> str:
    """Pre-flight ``exiftool -ver`` once per path per process.

    PyExifTool blocks forever reading from a child that died at startup
    (endless "Separating files" spinner, uncancellable job; Linux beta
    report 2026-07-06). Failing here instead surfaces the actual stderr
    of the broken spawn.
    """
    if path in _verified:
        return path
    try:
        result = subprocess.run(
            [path, "-ver"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(
            f"exiftool at {path} cannot be started: {e}"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(
            f"exiftool at {path} failed to start (exit "
            f"{result.returncode}):\n{result.stderr.strip()}"
        )
    _verified.add(path)
    return path


def _ensure_env_on_path(*dirs: str) -> None:
    """Prepend ``dirs`` to this process's PATH (idempotent)."""
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    missing = [d for d in dirs if d not in parts]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, *parts])
