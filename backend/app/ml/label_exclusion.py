"""
Label exclusion: zero-out excluded labels and renormalize confidences.

Applies label exclusion by zeroing out confidence for excluded class IDs,
renormalizing remaining confidences to sum to 1.0, and re-sorting by confidence.

JSON files on disk remain untouched as raw ground truth. Exclusion is applied
in-memory before writing to the database or passing to smoothing.
"""

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Non-label classes that should always be excluded from classifications.
# These are generic labels from the model that aren't real labels and should
# never appear as label predictions in the UI or affect smoothing/rollup.
# Stripped during JSON ingest and before postprocessing -- rollup and smoothing
# never see them. Add new junk classes here (e.g. "calibration", "setup").
NON_LABEL_CLASSES = frozenset({"bait", "blank", "empty", "false detection", "none"})


def filter_classifications(
    classifications: list[list],
    excluded_class_ids: set[str],
) -> list[list]:
    """
    Zero out excluded labels and renormalize remaining confidences.

    Args:
        classifications: List of [class_id, confidence] pairs
        excluded_class_ids: Set of class IDs to exclude

    Returns:
        New list of [class_id, confidence] sorted by confidence descending,
        with excluded labels removed and remaining confidences renormalized
        to sum to 1.0. Returns empty list if no labels remain.
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


def build_excluded_class_ids(
    class_categories: dict[str, str],
    excluded_labels: list[str] | None = None,
) -> set[str]:
    """
    Build the full set of class IDs to exclude.

    Always includes NON_LABEL_CLASSES (blank, empty, false detection, none).
    Additionally includes any user-configured excluded labels.

    Args:
        class_categories: Mapping of class_id -> class_name from JSON
        excluded_labels: Optional user-configured label names to exclude

    Returns:
        Set of class ID strings to exclude
    """
    if not class_categories:
        return set()

    # Build name -> [class_ids] lookup (lowercase for NON_LABEL_CLASSES matching)
    name_to_ids: dict[str, list[str]] = {}
    name_lower_to_ids: dict[str, list[str]] = {}
    for cls_id, name in class_categories.items():
        name_to_ids.setdefault(name, []).append(cls_id)
        name_lower_to_ids.setdefault(name.lower(), []).append(cls_id)

    excluded_class_ids: set[str] = set()

    # Always exclude non-label classes (case-insensitive)
    for non_label in NON_LABEL_CLASSES:
        for cls_id in name_lower_to_ids.get(non_label, []):
            excluded_class_ids.add(str(cls_id))

    # Exclude user-configured labels (exact match)
    if excluded_labels:
        for label_name in excluded_labels:
            for cls_id in name_to_ids.get(label_name, []):
                excluded_class_ids.add(str(cls_id))

    return excluded_class_ids


def apply_label_exclusion_to_results(
    md_results: dict,
    excluded_labels: list[str] | None = None,
) -> dict:
    """
    Apply label exclusion to a full MegaDetector JSON results dict (in place).

    Always excludes NON_LABEL_CLASSES (blank, empty, false detection, none).
    Additionally excludes any user-configured labels.

    Args:
        md_results: Full MegaDetector JSON dict (modified in place)
        excluded_labels: Optional list of label names to exclude

    Returns:
        The modified dict (same reference as input)
    """
    class_categories = md_results.get("classification_categories", {})
    excluded_class_ids = build_excluded_class_ids(class_categories, excluded_labels)

    if not excluded_class_ids:
        return md_results

    for img in md_results.get("images", []):
        for det in img.get("detections", []):
            if "classifications" in det and det["classifications"]:
                det["classifications"] = filter_classifications(
                    det["classifications"], excluded_class_ids
                )

    return md_results
