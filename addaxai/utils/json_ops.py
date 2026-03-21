"""JSON recognition-file manipulation for AddaxAI.

Functions to read, modify, and merge MegaDetector/classifier JSON output files.
No UI or heavy dependencies — only stdlib.
"""

import json
import os
from typing import Any, Dict, Optional


def fetch_label_map_from_json(path_to_json: str) -> Dict[str, str]:
    """Read and return the detection_categories dict from a recognition JSON."""
    with open(path_to_json, "r") as f:
        data = json.load(f)
    return data['detection_categories']


def check_json_paths(path_to_json: str, base_folder: str) -> str:
    """Determine whether image paths in a recognition JSON are absolute or relative.

    Args:
        path_to_json: Path to the recognition JSON file.
        base_folder: The base folder to check absolute paths against.

    Returns:
        "absolute" or "relative".
    """
    with open(path_to_json, "r") as f:
        data = json.load(f)
    path = os.path.normpath(data['images'][0]['file'])
    if path.startswith(os.path.normpath(base_folder)):
        return "absolute"
    else:
        return "relative"


def make_json_relative(path_to_json: str, base_folder: str) -> None:
    """Convert absolute image paths in a recognition JSON to relative paths.

    Args:
        path_to_json: Path to the recognition JSON file.
        base_folder: The base folder to strip from absolute paths.
    """
    if check_json_paths(path_to_json, base_folder) == "absolute":
        with open(path_to_json, "r") as f:
            data = json.load(f)

        for image in data['images']:
            absolute_path = os.path.normpath(image['file'])
            relative_path = absolute_path.replace(os.path.normpath(base_folder), "")[1:]
            image['file'] = relative_path

        with open(path_to_json, "w") as f:
            json.dump(data, f, indent=1)


def make_json_absolute(path_to_json: str, base_folder: str) -> None:
    """Convert relative image paths in a recognition JSON to absolute paths.

    Args:
        path_to_json: Path to the recognition JSON file.
        base_folder: The base folder to prepend to relative paths.
    """
    if check_json_paths(path_to_json, base_folder) == "relative":
        with open(path_to_json, "r") as f:
            data = json.load(f)

        for image in data['images']:
            relative_path = image['file']
            absolute_path = os.path.normpath(os.path.join(base_folder, relative_path))
            image['file'] = absolute_path

        with open(path_to_json, "w") as f:
            json.dump(data, f, indent=1)


def append_to_json(path_to_json: str, object_to_be_appended: Dict[str, Any]) -> None:
    """Add key-value pairs to the 'info' section of a recognition JSON."""
    with open(path_to_json, "r") as f:
        data = json.load(f)

    data['info'].update(object_to_be_appended)

    with open(path_to_json, "w") as f:
        json.dump(data, f, indent=1)


def change_hitl_var_in_json(path_to_json: str, status: str) -> None:
    """Set the human-in-the-loop status in a recognition JSON.

    Args:
        path_to_json: Path to the recognition JSON file.
        status: New HITL status string (e.g. "in-progress", "completed").
    """
    with open(path_to_json, "r") as f:
        data = json.load(f)

    data['info']["addaxai_metadata"]["hitl_status"] = status

    with open(path_to_json, "w") as f:
        json.dump(data, f, indent=1)


def get_hitl_var_in_json(path_to_json: str) -> str:
    """Read the human-in-the-loop status from a recognition JSON.

    Supports both 'addaxai_metadata' and legacy 'ecoassist_metadata' keys.

    Returns:
        The HITL status string, or "never-started" if not set.
    """
    with open(path_to_json, "r") as f:
        data = json.load(f)
    metadata = data['info'].get("addaxai_metadata")
    if metadata is None:
        metadata = data['info'].get("ecoassist_metadata")

    if metadata and "hitl_status" in metadata:
        return metadata["hitl_status"]
    return "never-started"


def merge_jsons(image_json: Optional[str], video_json: Optional[str], output_file_path: str) -> None:
    """Merge separate image and video recognition JSONs into a single file.

    Args:
        image_json: Path to the image recognition JSON, or None.
        video_json: Path to the video recognition JSON, or None.
        output_file_path: Path to write the merged output JSON.
    """
    image_data = None
    video_data = None

    if image_json:
        with open(image_json, 'r') as f:
            image_data = json.load(f)

    if video_json:
        with open(video_json, 'r') as f:
            video_data = json.load(f)

    # Pick the source that exists (prefer image if both)
    source = image_data if image_data else video_data

    if image_data and video_data:
        merged_images = image_data['images'] + video_data['images']
    elif image_data:
        merged_images = image_data['images']
    else:
        merged_images = video_data['images']

    merged_data = {
        "images": merged_images,
        "detection_categories": source['detection_categories'],
        "info": source['info'],
        "classification_categories": source.get('classification_categories', {}),
        "forbidden_classes": source.get('forbidden_classes', {}),
    }

    with open(output_file_path, 'w') as f:
        json.dump(merged_data, f, indent=1)
