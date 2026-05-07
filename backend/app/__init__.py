"""AddaxAI Backend Application."""

import sys
from pathlib import Path


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
