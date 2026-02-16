"""
Species exclusion: zero-out excluded species and renormalize confidences.

Applies species exclusion by zeroing out confidence for excluded class IDs,
renormalizing remaining confidences to sum to 1.0, and re-sorting by confidence.

JSON files on disk remain untouched as raw ground truth. Exclusion is applied
in-memory before writing to the database or passing to smoothing.
"""

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def filter_classifications(
    classifications: list[list],
    excluded_class_ids: set[str],
) -> list[list]:
    """
    Zero out excluded species and renormalize remaining confidences.

    Args:
        classifications: List of [class_id, confidence] pairs
        excluded_class_ids: Set of class IDs to exclude

    Returns:
        New list of [class_id, confidence] sorted by confidence descending,
        with excluded species removed and remaining confidences renormalized
        to sum to 1.0. Returns empty list if no species remain.
    """
    if not classifications or not excluded_class_ids:
        return classifications

    remaining = [
        [cls_id, conf]
        for cls_id, conf in classifications
        if str(cls_id) not in excluded_class_ids
    ]

    if not remaining:
        return []

    total = sum(conf for _, conf in remaining)
    if total <= 0:
        return []

    renormalized = [
        [cls_id, round(conf / total, 5)]
        for cls_id, conf in remaining
    ]
    renormalized.sort(key=lambda x: x[1], reverse=True)

    return renormalized


def apply_species_exclusion_to_results(
    md_results: dict,
    excluded_species: list[str],
) -> dict:
    """
    Apply species exclusion to a full MegaDetector JSON results dict (in place).

    Builds an excluded class ID set from classification_categories, then calls
    filter_classifications() on every detection.

    Args:
        md_results: Full MegaDetector JSON dict (modified in place)
        excluded_species: List of species names to exclude

    Returns:
        The modified dict (same reference as input)
    """
    if not excluded_species:
        return md_results

    class_categories = md_results.get("classification_categories", {})
    if not class_categories:
        return md_results

    # Build set of class IDs to exclude (name -> id lookup)
    name_to_ids: dict[str, list[str]] = {}
    for cls_id, name in class_categories.items():
        name_to_ids.setdefault(name, []).append(cls_id)

    excluded_class_ids: set[str] = set()
    for species_name in excluded_species:
        for cls_id in name_to_ids.get(species_name, []):
            excluded_class_ids.add(str(cls_id))

    if not excluded_class_ids:
        return md_results

    for img in md_results.get("images", []):
        for det in img.get("detections", []):
            if "classifications" in det and det["classifications"]:
                det["classifications"] = filter_classifications(
                    det["classifications"], excluded_class_ids
                )

    return md_results
