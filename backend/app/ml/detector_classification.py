"""Helpers for reusing custom detector labels as classification results."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def add_detector_classes_as_classifications(document: dict[str, Any]) -> dict[str, Any]:
    """Attach each custom detector's class/confidence to its detection record.

    Detection category IDs and names become classification IDs and names in
    the same JSON document. This is deliberately a pure mapping step: it does
    not crop images, load a classifier, or change the detector's confidence.
    """
    raw_categories = document.get("detection_categories")
    if not isinstance(raw_categories, dict) or not raw_categories:
        raise ValueError("Detection results must declare detection_categories")

    categories: dict[str, str] = {}
    for category_id, name in raw_categories.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Detection category names must be non-empty strings")
        categories[str(category_id)] = name

    images = document.get("images")
    if not isinstance(images, list):
        raise ValueError("Detection results must declare an images list")

    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Detection image entries must be objects")
        detections = image.get("detections") or []
        if not isinstance(detections, list):
            raise ValueError("Detection entries must be a list or null")
        for detection in detections:
            if not isinstance(detection, dict):
                raise ValueError("Detection entries must be objects")
            category_id = str(detection.get("category"))
            if category_id not in categories:
                raise ValueError(
                    f"Detection category id {category_id!r} has no declared class name"
                )
            confidence = detection.get("conf")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, int | float)
                or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1
            ):
                raise ValueError("Detection confidence must be a finite number from 0 to 1")
            # Preserve category IDs (including zero) and the confidence value
            # as emitted by the detector. Re-run after resume to repair any
            # cached JSON that predates this reuse mapping.
            detection["classifications"] = [[category_id, confidence]]

    document["classification_categories"] = categories
    return document


def reuse_detector_classes_in_json(path: Path) -> None:
    """Add classifier fields to a detector JSON file with an atomic replace."""
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    add_detector_classes_as_classifications(document)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = output.name
            json.dump(document, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
