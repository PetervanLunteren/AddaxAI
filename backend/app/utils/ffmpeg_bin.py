"""
Locate the ffmpeg binary for an analysis environment.

The tracking script decodes videos through an ffmpeg pipe rather than
OpenCV, so it gets the platform's hardware decoder (VideoToolbox on
macOS, D3D11 on Windows, VAAPI on Linux) and a downscale in the same
pass. The binary ships inside the environment as a conda-forge package,
like exiftool (see `exiftool_bin.py`), because the installers bundle
nothing and fresh machines have no ffmpeg on PATH.

Resolution order: the env's own binary, then PATH (dev machines, CI).
Raises RuntimeError when neither exists so the run fails before a
single frame is decoded, with a message that says what to rebuild.
"""

import os
import shutil

from app.core.config import get_settings


def resolve_ffmpeg(env_name: str) -> str:
    """Absolute path to ffmpeg for ``env-<env_name>``."""
    env_dir = get_settings().user_data_dir / "envs" / f"env-{env_name}"
    candidate = (
        env_dir / "Library" / "bin" / "ffmpeg.exe"
        if os.name == "nt"
        else env_dir / "bin" / "ffmpeg"
    )
    if candidate.is_file():
        return str(candidate)

    on_path = shutil.which("ffmpeg")
    if on_path is not None:
        return on_path

    raise RuntimeError(
        f"ffmpeg not found. Expected it inside the analysis environment "
        f"({env_dir}) or on PATH. Rebuild that environment from Settings."
    )
