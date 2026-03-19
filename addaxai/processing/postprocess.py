"""Postprocessing utilities for AddaxAI.

File separation by detection type, confidence-based sorting, and
related helpers. The main postprocess() orchestrator remains in
AddaxAI_GUI.py for now (too tightly coupled to UI/globals).
"""

import math
import os
import shutil
from pathlib import Path


# Confidence bucket directory names for sorting detections by confidence
CONF_DIRS = {
    0.0: "conf_0.0",
    0.1: "conf_0.0-0.1",
    0.2: "conf_0.1-0.2",
    0.3: "conf_0.2-0.3",
    0.4: "conf_0.3-0.4",
    0.5: "conf_0.4-0.5",
    0.6: "conf_0.5-0.6",
    0.7: "conf_0.6-0.7",
    0.8: "conf_0.7-0.8",
    0.9: "conf_0.8-0.9",
    1.0: "conf_0.9-1.0",
}


def format_size(size):
    """Format a byte count as a human-readable string (B, KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{round(size)} {unit}"
        size /= 1024.0


def move_files(file, detection_type, file_placement, max_detection_conf,
               sep_conf, dst_root, src_dir, manually_checked):
    """Move or copy a file into a subdirectory based on detection results.

    Args:
        file: Relative path of the file within src_dir.
        detection_type: Category name (e.g. "animal", "person", "empty").
        file_placement: 1 = move, 2 = copy.
        max_detection_conf: Highest detection confidence for this file.
        sep_conf: Whether to add confidence-based subdirectories.
        dst_root: Root destination directory.
        src_dir: Source directory containing the file.
        manually_checked: Whether the file was human-verified.

    Returns:
        The new relative file path within dst_root.
    """
    if sep_conf and detection_type != "empty":
        if manually_checked:
            confidence_dir = "verified"
        else:
            ceiled_confidence = math.ceil(max_detection_conf * 10) / 10.0
            confidence_dir = CONF_DIRS[ceiled_confidence]
        new_file = os.path.join(detection_type, confidence_dir, file)
    else:
        new_file = os.path.join(detection_type, file)

    src = os.path.join(src_dir, file)
    dst = os.path.join(dst_root, new_file)

    Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)

    if file_placement == 1:
        shutil.move(src, dst)
    elif file_placement == 2:
        shutil.copy2(src, dst)

    return new_file
