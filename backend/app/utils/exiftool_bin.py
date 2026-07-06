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
"""

import shutil

from app.core.config import get_settings


def resolve_exiftool() -> str:
    """Return the absolute path to the exiftool binary."""
    env_dir = get_settings().user_data_dir / "envs" / "env-addaxai-base"
    candidates = (
        env_dir / "bin" / "exiftool",      # linux / macOS
        env_dir / "bin" / "exiftool.bat",  # windows (conda-forge layout)
    )
    for candidate in candidates:
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
