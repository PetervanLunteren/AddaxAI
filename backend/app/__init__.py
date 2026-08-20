"""AddaxAI Backend Application."""

import os
import sys
from pathlib import Path

from PIL import ImageFile

# Decode what a truncated image does contain instead of refusing it.
#
# Camera traps produce partly-written JPEGs routinely: a battery dies
# mid-write, an SD card is pulled, a copy is interrupted. Pillow's
# default is to raise `OSError: broken data stream when reading image
# file` on those, so every surface that decodes pixels (thumbnails,
# crops, filmstrips, the annotated copies, the EXIF date read) fails on
# a file MegaDetector itself reads without complaint: it sets this same
# flag in `megadetector/visualization/visualization_utils.py`, which
# `run_detector_batch` loads every image through. So the detector saw
# these files and we could not, which is the wrong way round.
#
# Measured on a real 2,281-file deployment: 24 files (1.1%) were
# affected, every one of them returning a 500 from the thumbnail
# endpoint.
#
# The trade is that a truncated file now yields a partial image, with
# flat grey where the data ran out, rather than an error. Cutting a test
# JPEG at 25 / 50 / 75 / 90% of its bytes recovers 28 / 52 / 78 / 91% of
# the frame, so in the ordinary case this is most of the picture. At the
# extreme it is not: those 24 files were cut so early that under 5% of
# each frame decodes, and they are grey whatever we do.
#
# Set here so it applies process-wide before any submodule imports
# Pillow. The inference scripts under `app/ml/inference/` run as
# standalone subprocesses in their own conda environments and cannot
# import `app`, so they each set it themselves.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# All AddaxAI env vars carry the ADDAXAI_ prefix, including the three
# HuggingFace endpoint settings. But huggingface_hub reads its own
# unprefixed names (HF_ENDPOINT, HF_HUB_DISABLE_XET, HF_TOKEN), the
# first of them at import time, so the prefixed values are copied over
# here, before any submodule can import huggingface_hub. The prefixed
# name wins when both are set, same as in Settings.
for _hf_name in ("HF_ENDPOINT", "HF_HUB_DISABLE_XET", "HF_TOKEN"):
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
