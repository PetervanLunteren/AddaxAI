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

Windows is special. The conda-forge win-64 package ships two entry
points in ``bin/``: ``exiftool`` (the perl script, not directly
executable on Windows: spawning it raises WinError 193) and
``exiftool.bat`` (a pl2bat wrapper that invokes plain ``perl``). The
wrapper only works when the env's perl is resolvable, which it is not
for our backend process because nothing ever activates the conda env.
So on Windows we return the .bat and prepend the env's binary dirs
(``Library/bin`` holds perl.exe) to this process's PATH; children
spawned by PyExifTool inherit it.
"""

import os
import shutil

from app.core.config import get_settings


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
            return str(candidate)
    else:
        candidate = env_dir / "bin" / "exiftool"
        if candidate.is_file():
            return str(candidate)

    on_path = shutil.which("exiftool")
    if on_path is not None:
        return on_path

    raise RuntimeError(
        "exiftool not found. Expected it inside the analysis environment "
        f"({env_dir}) or on PATH. Re-run the initial setup to rebuild the "
        "analysis environment."
    )


def _ensure_env_on_path(*dirs: str) -> None:
    """Prepend ``dirs`` to this process's PATH (idempotent)."""
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    missing = [d for d in dirs if d not in parts]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, *parts])
