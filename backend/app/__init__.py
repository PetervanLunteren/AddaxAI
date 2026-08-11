"""AddaxAI Backend Application."""

import os
import sys
from pathlib import Path

# All AddaxAI env vars carry the ADDAXAI_ prefix, including the two
# HuggingFace mirror settings documented for mainland China. But
# huggingface_hub reads its own unprefixed names (HF_ENDPOINT,
# HF_HUB_DISABLE_XET) at import time, so the prefixed values are copied
# over here, before any submodule can import huggingface_hub. The
# prefixed name wins when both are set, same as in Settings.
for _hf_name in ("HF_ENDPOINT", "HF_HUB_DISABLE_XET"):
    _value = os.environ.get(f"ADDAXAI_{_hf_name}", "").strip()
    if _value:
        os.environ[_hf_name] = _value


def _read_version() -> str:
    """
    Resolve the canonical app version from the repo-root `VERSION` file.

    Looked up in two places:
      1. The PyInstaller bundle root (`sys._MEIPASS / VERSION`) when
         we're running as a packaged binary. The spec file copies
         `VERSION` into `_MEIPASS` at build time.
      2. The dev tree at `<repo>/VERSION` when running uvicorn / pytest
         directly (`<repo>/backend/app/__init__.py` -> parents[2] is
         the repo root).

    Returns "0.0.0+unknown" if neither file is readable so a missing
    bundle never silently lies about the version.
    """
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "VERSION")
    candidates.append(Path(__file__).resolve().parents[2] / "VERSION")
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
    return "0.0.0+unknown"


__version__ = _read_version()
