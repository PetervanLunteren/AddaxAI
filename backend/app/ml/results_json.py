"""Streaming readers for MegaDetector-format results JSON.

``json.load`` on a results file peaks at roughly 6.4x the file size in RAM
(measured), so a large deployment (hundreds of thousands of images) can OOM
the backend. These helpers walk the file with ``ijson`` at flat memory
instead (~1 MB regardless of size).

Numbers are parsed with ``use_float=True`` so they come back as ``float``,
exactly like ``json.load``. ijson's default is ``decimal.Decimal``, which
would break Float columns, downstream arithmetic, and JSON re-serialization
of the ``exif_metadata`` blob.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import-untyped]


def iter_images(json_path: Path) -> Iterator[dict[str, Any]]:
    """Yield each entry of the top-level ``images`` array, one at a time."""
    with open(json_path, "rb") as f:
        yield from ijson.items(f, "images.item", use_float=True)


def read_top_level_object(json_path: Path, key: str) -> dict[str, Any]:
    """Return a top-level object value (e.g. ``classification_categories``).

    Returns ``{}`` when the key is absent or is not an object. Only that
    object is held in memory. Cheap when the key precedes ``images`` in the
    file (the merge writer puts metadata first); for older files where it
    follows ``images`` ijson still finds it, at flat memory.
    """
    with open(json_path, "rb") as f:
        for value in ijson.items(f, key, use_float=True):
            return value if isinstance(value, dict) else {}
    return {}
