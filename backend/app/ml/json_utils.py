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

from app.utils.fs_hidden import mkdir_hidden_addaxai


def collect_md_failures(md_results: dict) -> list[dict]:
    """Return one entry per failed image/video in a MegaDetector JSON.

    MegaDetector's `process_video` writes entries shaped like
    ``{"file": "...", "failure": "...", "detections": null}`` when a
    video can't be decoded (corrupt file, unsupported codec, no
    retrievable frames). Naive iteration over `detections` blows up
    with `TypeError: 'NoneType' object is not iterable`. Use this
    helper at the start of the pipeline to extract those entries for
    user-facing warnings, and use the `iter_*` helpers below to walk
    the JSON without tripping on them.
    """
    return [
        {"file": img.get("file"), "reason": img.get("failure")}
        for img in (md_results.get("images") or [])
        if img.get("failure")
    ]


def extract_animal_detections(
    md_results: dict, *, min_confidence: float
) -> list[tuple[int, int, dict]]:
    """
    Extract animal detections with their indices for classification.

    ``min_confidence`` is the project's classification gate: MegaDetector
    runs at its 0.01 output cap, so the JSON carries a long
    near-noise tail that must not be classified. Detections below the
    gate stay in the JSON and the database as raw animal boxes; they
    are just not sent to the classifier.

    Args:
        md_results: MegaDetector JSON results dict
        min_confidence: gate below which animal detections are skipped

    Returns:
        List of (image_index, detection_index, detection_dict) tuples.
        Only detections with category == "1" (animal) at or above the
        gate.
    """
    animals: list[tuple[int, int, dict]] = []

    # Failure entries have `detections: null` (see collect_md_failures).
    # Skip them so this loop does not crash on a corrupt-video JSON.
    for img_idx, img in enumerate(md_results.get("images") or []):
        if img.get("failure"):
            continue
        for det_idx, det in enumerate(img.get("detections") or []):
            if det.get("category") != "1":  # animal only
                continue
            if float(det.get("conf", 0.0)) < min_confidence:
                continue
            animals.append((img_idx, det_idx, det))

    return animals


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
    mkdir_hidden_addaxai(artifacts)
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
    for img in md_results.get("images") or []:
        # Assign file_id if not present
        if "file_id" not in img:
            img["file_id"] = str(uuid.uuid4())

        # Assign detection_id to each detection. Tolerant of failure
        # entries where `detections` is null (process_video pattern).
        for det in img.get("detections") or []:
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

    with open(taxonomy_csv_path, newline="", encoding="utf-8") as f:
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


def trim_classification_results(
    md_results: dict,
    *,
    max_classifications: int = 5,
) -> int:
    """
    Trim classification results to top-N per detection in-place.

    Truncates each detection's classifications list to the top
    max_classifications entries (already sorted descending by confidence).
    Prunes classification_categories and classification_category_descriptions
    to only include class IDs still referenced by at least one detection.

    Args:
        md_results: MegaDetector/AddaxAI JSON results dict (modified in-place)
        max_classifications: Maximum classification entries to keep per
            detection. Defaults to 5 (matches SpeciesNet API and the
            rollup algorithm in taxonomic_rollup.py).

    Returns:
        Number of class IDs removed from classification_categories.
    """
    categories = md_results.get("classification_categories")
    if not categories:
        return 0

    original_count = len(categories)

    # Trim each detection and collect referenced class IDs. Tolerant of
    # `detections: null` entries written by process_video for videos it
    # could not decode.
    referenced_ids: set[str] = set()
    for img in md_results.get("images") or []:
        for det in img.get("detections") or []:
            cls_list = det.get("classifications")
            if not cls_list:
                continue
            det["classifications"] = [
                [class_id, round(conf, 5)]
                for class_id, conf in cls_list[:max_classifications]
            ]
            for class_id, _conf in det["classifications"]:
                referenced_ids.add(str(class_id))

    # Prune classification_categories
    for key in list(categories):
        if key not in referenced_ids:
            del categories[key]

    # Prune classification_category_descriptions
    descriptions = md_results.get("classification_category_descriptions")
    if descriptions:
        for key in list(descriptions):
            if key not in referenced_ids:
                del descriptions[key]

    return original_count - len(categories)


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
