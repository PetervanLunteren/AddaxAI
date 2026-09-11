"""The class-mapping file a non-MegaDetector detector needs.

Some detectors cannot name their own classes to the megadetector package.
SharkTrack is the case: it is a plain ultralytics checkpoint whose only
class is ``elasmobranch``, and ``PTDetector`` otherwise asserts class
indices in ``{0,1,2}`` and adds one, so every shark would be stored as
``animal``.

The package's answer is a JSON file of class ids to names, given to
``run_detector_batch`` as ``--class_mapping_filename``. Its loader
(``_load_custom_class_mapping``) both switches the package to the model's
native class indices and makes that map the run's ``detection_categories``.
Its own words: "Allows the use of non-MD models, disables the code that
enforces MD-like class lists."

The map itself lives in the catalog (``ModelManifest.class_mapping``), so
there is one declaration per detector. This module is only the file: both
the image path and the video path write it the same way, so the two cannot
drift, and the ids a run reports are the ids the catalog declared.
"""

from __future__ import annotations

import json
from pathlib import Path

CLASS_MAPPING_FILENAME = "class_mapping.json"


def write_class_mapping(mapping: dict[str, str], directory: Path) -> Path:
    """Write ``mapping`` into ``directory`` and return the path.

    Callers pass the directory they already use for the run's file list,
    so the mapping is cleaned up with it: a temporary directory for the
    image path, the artifacts folder for the video path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CLASS_MAPPING_FILENAME
    path.write_text(json.dumps(mapping))
    return path
