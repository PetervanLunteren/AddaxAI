"""
Label exclusion and non-label skip logic.

Two separate concerns handled here:

1. **User exclusion** (filter + renormalize): labels the user marked as
   not present in their project area are removed from the classification
   list and remaining confidences renormalized to sum to 1.0. This
   changes which label is assigned to a detection.

2. **Non-label skip** (DB gatekeeper): if the top-1 prediction after
   user filtering is a NON_LABEL_CLASS (blank, bait, etc.), the
   detection is not loaded to the database at all. The bbox is treated
   as a false positive.

These two steps are independent. NON_LABEL_CLASSES never participate
in renormalization. JSON files on disk remain untouched as raw ground
truth.
"""

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Non-label classes: predictions that mean "nothing here" or "false positive".
# A detection whose top-1 is one of these is not loaded to the database.
# Also stripped before smoothing/rollup so they don't corrupt those algorithms.
# Add new junk classes here (e.g. "calibration", "setup").
NON_LABEL_CLASSES = frozenset({
    "bait", "blank", "empty", "false detection", "none", "vide",
})


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

    Always includes NON_LABEL_CLASSES (bait, blank, empty, false detection,
    none, vide). Additionally includes any user-configured excluded labels.

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


def build_user_excluded_class_ids(
    class_categories: dict[str, str],
    excluded_labels: list[str] | None = None,
) -> set[str]:
    """
    Build class IDs for user-excluded labels only.

    Does NOT include NON_LABEL_CLASSES. Used for filtering and
    renormalization during DB loading.

    Args:
        class_categories: Mapping of class_id -> class_name from JSON
        excluded_labels: User-configured label names to exclude

    Returns:
        Set of class ID strings to exclude
    """
    if not class_categories or not excluded_labels:
        return set()

    name_to_ids: dict[str, list[str]] = {}
    for cls_id, name in class_categories.items():
        name_to_ids.setdefault(name, []).append(cls_id)

    excluded: set[str] = set()
    for label_name in excluded_labels:
        for cls_id in name_to_ids.get(label_name, []):
            excluded.add(str(cls_id))

    return excluded


def build_non_label_class_ids(
    class_categories: dict[str, str],
) -> set[str]:
    """
    Build class IDs for NON_LABEL_CLASSES only.

    Used for the skip decision during DB loading: if a detection's
    top-1 prediction (after user filtering) is one of these, the
    detection is not loaded.

    Args:
        class_categories: Mapping of class_id -> class_name from JSON

    Returns:
        Set of class ID strings for non-label classes
    """
    if not class_categories:
        return set()

    non_label_ids: set[str] = set()
    for cls_id, name in class_categories.items():
        if name.lower() in NON_LABEL_CLASSES:
            non_label_ids.add(str(cls_id))

    return non_label_ids


def should_skip_detection(
    det: dict,
    user_excluded_class_ids: set[str],
    non_label_class_ids: set[str],
) -> bool:
    """
    Return True if a detection should not be loaded to the database.

    Steps:
    1. If no classifications, don't skip (unclassified animal).
    2. Apply user exclusion filter (remove + renormalize).
    3. If nothing remains after user filtering, skip.
    4. If filtered top-1 is a NON_LABEL class, skip (false positive bbox).

    Args:
        det: Detection dict from JSON
        user_excluded_class_ids: Class IDs from user's excluded_classes
        non_label_class_ids: Class IDs for NON_LABEL_CLASSES

    Returns:
        True if detection should be skipped, False if it should be loaded.
    """
    raw = det.get("classifications")
    if not raw:
        return False

    # Apply user exclusions
    if user_excluded_class_ids:
        filtered = filter_classifications(raw, user_excluded_class_ids)
    else:
        filtered = raw

    # Nothing left after user filtering
    if not filtered:
        return True

    # Top-1 is a non-label class (blank, bait, etc.) → false positive
    top_class_id = str(filtered[0][0])
    return top_class_id in non_label_class_ids


def is_non_label_detection(
    det: dict,
    excluded_class_ids: set[str],
) -> bool:
    """
    Return True if a detection should be skipped (not loaded to DB).

    A detection is skipped when:
    1. It HAS classifications (went through a classifier), AND
    2. After filtering out excluded/non-label class IDs, no classifications
       remain.

    Detections without any classifications (unclassified animals) are NOT
    skipped. Non-animal detections (person, vehicle) never have
    classifications, so they are never skipped.

    Args:
        det: Detection dict from JSON (has "classifications" key if classified)
        excluded_class_ids: Set of class IDs to exclude

    Returns:
        True if detection should be skipped, False if it should be loaded.
    """
    if not excluded_class_ids:
        return False

    raw_classifications = det.get("classifications")
    if not raw_classifications:
        return False

    filtered = filter_classifications(raw_classifications, excluded_class_ids)
    return len(filtered) == 0


def apply_label_exclusion_to_results(
    md_results: dict,
    excluded_labels: list[str] | None = None,
) -> dict:
    """
    Apply label exclusion to a full MegaDetector JSON results dict (in place).

    Always excludes NON_LABEL_CLASSES (bait, blank, empty, false detection,
    none, vide). Additionally excludes any user-configured labels.

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
