"""
JSON utilities for MegaDetector and AddaxAI format handling.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Clean, testable functions

Created by Claude Code on 2026-01-05
"""

import csv
import uuid
from pathlib import Path
from typing import Any


def extract_animal_detections(md_results: dict) -> list[tuple[int, int, dict]]:
    """
    Extract animal detections with their indices for classification.

    Args:
        md_results: MegaDetector JSON results dict

    Returns:
        List of (image_index, detection_index, detection_dict) tuples
        Only includes detections where category == "1" (animal)
    """
    animals: list[tuple[int, int, dict]] = []

    for img_idx, img in enumerate(md_results.get("images", [])):
        for det_idx, det in enumerate(img.get("detections", [])):
            if det.get("category") == "1":  # animal
                animals.append((img_idx, det_idx, det))

    return animals


def build_addaxai_metadata(
    deployment_id: str,
    det_model_id: str,
    cls_model_id: str | None,
    md_results: dict,
) -> dict[str, Any]:
    """
    Build addaxai_metadata section for extended JSON.

    Args:
        deployment_id: Deployment UUID
        det_model_id: Detection model ID
        cls_model_id: Classification model ID (or None)
        md_results: MegaDetector results dict

    Returns:
        Metadata dict for addaxai_metadata section
    """
    return {
        "deployment_id": deployment_id,
        "selected_det_modelID": det_model_id,
        "selected_cls_modelID": cls_model_id,
        "n_images": len(md_results.get("images", [])),
        "n_videos": 0,
    }


def create_artifacts_folder(deployment_folder: Path) -> Path:
    """
    Create .addaxai artifacts folder in deployment directory.

    Args:
        deployment_folder: Path to deployment folder

    Returns:
        Path to .addaxai artifacts folder

    Raises:
        OSError: If folder creation fails
    """
    artifacts = deployment_folder / ".addaxai"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def assign_uuids_to_detection_json(md_results: dict) -> None:
    """
    Assign file_id and detection_id UUIDs to MegaDetector JSON in-place.

    Modifies md_results dict by adding:
    - "file_id" to each image
    - "detection_id" to each detection

    Args:
        md_results: MegaDetector JSON results dict (modified in-place)
    """
    for img in md_results.get("images", []):
        # Assign file_id if not present
        if "file_id" not in img:
            img["file_id"] = str(uuid.uuid4())

        # Assign detection_id to each detection
        for det in img.get("detections", []):
            if "detection_id" not in det:
                det["detection_id"] = str(uuid.uuid4())


def format_class_names_for_json(class_names: dict[str, str]) -> dict[str, str]:
    """
    Format class names dictionary for JSON output.

    Ensures all keys are strings (not integers) for JSON compatibility.

    Args:
        class_names: Dict mapping class ID to class name

    Returns:
        Dict with string keys
    """
    return {str(k): v for k, v in class_names.items()}


def build_classification_category_descriptions(
    classification_categories: dict[str, str],
    taxonomy_csv_path: Path,
) -> dict[str, str]:
    """
    Build classification_category_descriptions from taxonomy.csv.

    Maps each classification category ID to a 7-token taxonomy string:
    ``common_name;class;order;family;genus;species;common_name`` (all lowercase).

    MegaDetector's smoothing code expects 7-token strings where token 0 is an
    identifier, tokens 1-5 are the taxonomy (class through species), and
    token 6 is the display name.

    Args:
        classification_categories: Dict mapping class_id -> class_name
            (e.g. {"0": "fox", "1": "deer"})
        taxonomy_csv_path: Path to taxonomy.csv with columns:
            model_class,class,order,family,genus,species

    Returns:
        Dict mapping class_id -> 7-token taxonomy string
        (e.g. {"0": "fox;mammalia;carnivora;canidae;vulpes;vulpes;fox"})
    """
    # Load taxonomy CSV into lookup by model_class name
    taxonomy_lookup: dict[str, dict[str, str]] = {}

    with open(taxonomy_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_class = row.get("model_class", "").strip().lower()
            if not model_class:
                continue
            taxonomy_lookup[model_class] = {
                "class": row.get("class", "").strip().lower(),
                "order": row.get("order", "").strip().lower(),
                "family": row.get("family", "").strip().lower(),
                "genus": row.get("genus", "").strip().lower(),
                "species": row.get("species", "").strip().lower(),
            }

    # Map each classification category to its 7-token taxonomy string
    descriptions: dict[str, str] = {}
    for class_id, class_name in classification_categories.items():
        name_lower = class_name.strip().lower()
        if name_lower in taxonomy_lookup:
            tax = taxonomy_lookup[name_lower]
            # 7-token format: name;class;order;family;genus;species;name
            tokens = [
                name_lower,
                tax["class"],
                tax["order"],
                tax["family"],
                tax["genus"],
                tax["species"],
                name_lower,
            ]
            descriptions[str(class_id)] = ";".join(tokens)

    return descriptions


def get_relative_path(absolute_path: Path, base_folder: Path) -> str:
    """
    Get relative path from base folder to absolute path.

    Args:
        absolute_path: Absolute path to file
        base_folder: Base folder path

    Returns:
        Relative path as string

    Raises:
        ValueError: If absolute_path is not under base_folder
    """
    try:
        return str(absolute_path.relative_to(base_folder))
    except ValueError as e:
        raise ValueError(
            f"Path {absolute_path} is not under base folder {base_folder}"
        ) from e
