"""
Build a synthetic MegaDetector-format JSON for full-image classifiers.

Full-image classifiers (e.g. AHDRIFT-v1) label the whole frame with a
single class and do not need a detector to run first. The downstream
classification pipeline still expects detections in MegaDetector's
standard JSON shape, so this module synthesises one full-image detection
per image (bbox covering the entire frame, conf 1.0, category "1" /
animal). Once written, the JSON flows through the regular Phase 4 - 8
path unchanged: classify, merge, DB load, postprocessing, embedding.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import ExifTags, Image

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# DateTimeOriginal lives in the Exif SubIFD, reachable via the IFD pointer
# at tag 0x8769 in the base IFD. Resolve the tag id once at import time.
_EXIF_IFD_POINTER = 0x8769
_DATETIME_ORIGINAL_TAG = next(
    tag_id for tag_id, name in ExifTags.TAGS.items()
    if name == "DateTimeOriginal"
)


def _read_image_metadata(
    image_path: Path,
) -> tuple[int, int, str | None]:
    """
    Return (width, height, DateTimeOriginal-or-None) for one image.

    DateTimeOriginal is normalised to MD's wire format
    ("YYYY:MM:DD HH:MM:SS"). When EXIF is missing or unreadable the
    timestamp is None and the JSON pipeline's pre-flight pass will skip
    the file (same path as a corrupt-EXIF image would take coming out
    of MegaDetector).
    """
    with Image.open(image_path) as img:
        width, height = img.size
        exif = img.getexif()
        dto: str | None = None
        if exif:
            sub = exif.get_ifd(_EXIF_IFD_POINTER)
            raw = sub.get(_DATETIME_ORIGINAL_TAG) if sub else None
            if isinstance(raw, str):
                cleaned = raw.strip("\x00").strip()
                if cleaned:
                    dto = cleaned
    return width, height, dto


def synthesize_full_image_json(
    image_paths: list[Path],
    deployment_folder: Path,
    output_path: Path,
) -> None:
    """
    Write a MegaDetector-shaped detection JSON with one full-image
    detection per input image.

    Each entry contains width, height, and exif_metadata
    (DateTimeOriginal) so that `_resolve_capture_timestamp` in
    `app.ml.json_pipeline` works without modification. Bbox is fixed at
    `[0, 0, 1, 1]`, category is "1" (animal), confidence is 1.0 — the
    synthetic detection is indistinguishable from a real animal
    detection to all downstream phases.
    """
    images_block: list[dict] = []
    for path in image_paths:
        try:
            width, height, dto = _read_image_metadata(path)
        except Exception as e:
            logger.warning(
                f"Could not read metadata for {path}: {e}"
            )
            continue
        relative = path.relative_to(deployment_folder)
        entry: dict = {
            "file": str(relative),
            "width": width,
            "height": height,
            "detections": [
                {
                    "category": "1",
                    "conf": 1.0,
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                }
            ],
        }
        if dto:
            entry["exif_metadata"] = {"DateTimeOriginal": dto}
        images_block.append(entry)

    payload = {
        "images": images_block,
        "detection_categories": {
            "1": "animal",
            "2": "person",
            "3": "vehicle",
        },
        "info": {
            "detection_completion_time": datetime.now(UTC).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "detector": "full_image_synthetic",
            "format_version": "",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Synthesised full-image detection JSON: "
        f"{len(images_block)} image(s) -> {output_path}"
    )
