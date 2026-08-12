"""
Label exclusion and non-label skip logic.

Two separate concerns handled here:

1. **User exclusion** (filter): labels the user marked as not present
   in their project area are removed from the classification list.
   Remaining confidences keep their raw values (no renormalization).
   This changes which label is assigned to a detection.

2. **Non-label skip** (DB gatekeeper): if the top-1 prediction after
   user filtering is a NON_LABEL_CLASS (blank, bait, etc.), the
   detection is not loaded to the database at all. The bbox is treated
   as a false positive.

These two steps are independent. JSON files on disk remain untouched
as raw ground truth.
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

# Non-wildlife classes: real detections that are not wild animals.
# Superset of NON_LABEL_CLASSES, adding every human and vehicle class
# name found across the model zoo. Used by wildlife-only statistics
# (the dashboard "Wildlife detected" chart). Matched case-insensitively
# against Detection/EventObservation labels. When a new model ships a
# class like "person" or "car", add it here.
NON_WILDLIFE_CLASSES = NON_LABEL_CLASSES | frozenset({
    "human", "homo_sapiens", "vehicle",
})


def filter_classifications(
    classifications: list[list],
    excluded_class_ids: set[str],
) -> list[list]:
    """
    Remove excluded labels from classifications.

    Remaining confidences keep their raw values (no renormalization).

    Args:
        classifications: List of [class_id, confidence] pairs
        excluded_class_ids: Set of class IDs to exclude

    Returns:
        New list of [class_id, confidence] sorted by confidence descending,
        with excluded labels removed. Returns empty list if no labels remain.
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

    remaining.sort(key=lambda x: x[1], reverse=True)
    return remaining


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
    non_label_class_ids: set[str],
) -> bool:
    """
    Return True if a detection should not be loaded to the database.

    Checks if the raw top-1 classification is a NON_LABEL class
    (blank, bait, etc.). User exclusion and rollup are handled
    separately in Phase 7 (postprocessing).

    Args:
        det: Detection dict from JSON
        non_label_class_ids: Class IDs for NON_LABEL_CLASSES

    Returns:
        True if detection should be skipped, False if it should be loaded.
    """
    raw = det.get("classifications")
    if not raw:
        return False

    top_class_id = str(raw[0][0])
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
    taxonomy_lookup: dict[str, dict[str, str]] | None = None,
) -> dict:
    """
    Apply label exclusion to a full MegaDetector JSON results dict (in place).

    Used in the postprocessing path (Phase 7).

    When taxonomy is available, this is a no-op: the classification lists
    stay untouched. Excluded species are handled by the geofence-aware
    rollup in apply_taxonomic_rollup_to_results(), which preserves the
    model's strong signal and redirects it to allowed ancestors (matching
    the official SpeciesNet API behavior).

    When taxonomy is NOT available (fallback), user-excluded and NON_LABEL
    classes are removed from classification lists.

    Note: the Phase 6 DB load path (json_pipeline.py) uses
    filter_and_rollup_classifications() directly for label assignment.

    Args:
        md_results: Full MegaDetector JSON dict (modified in place)
        excluded_labels: Optional list of label names to exclude
        taxonomy_lookup: Optional taxonomy lookup dict

    Returns:
        The modified dict (same reference as input)
    """
    if taxonomy_lookup:
        # Rollup handles excluded species with geofence awareness.
        # Do not filter here to preserve the full confidence landscape.
        return md_results

    class_categories = md_results.get("classification_categories", {})
    excluded_class_ids = build_excluded_class_ids(
        class_categories, excluded_labels
    )

    if not excluded_class_ids:
        return md_results

    # Iterate `images or []` / `detections or []` so failure entries from
    # process_video (corrupt video → `detections: null`) don't crash this.
    for img in md_results.get("images") or []:
        for det in img.get("detections") or []:
            if "classifications" not in det or not det["classifications"]:
                continue
            det["classifications"] = filter_classifications(
                det["classifications"], excluded_class_ids
            )

    return md_results


def strip_non_label_from_results(md_results: dict) -> dict:
    """
    Strip NON_LABEL classes from all detections in md_results (in place).

    Should be called AFTER taxonomic rollup but BEFORE smoothing, so that
    rollup sees the full confidence landscape (matching the official
    SpeciesNet API) while smoothing does not see blank/bait/etc.

    Args:
        md_results: Full MegaDetector JSON dict (modified in place)

    Returns:
        The modified dict (same reference as input)
    """
    class_categories = md_results.get("classification_categories", {})
    non_label_ids = build_non_label_class_ids(class_categories)
    if not non_label_ids:
        return md_results

    for img in md_results.get("images") or []:
        for det in img.get("detections") or []:
            if not det.get("classifications"):
                continue
            det["classifications"] = [
                [cls_id, conf]
                for cls_id, conf in det["classifications"]
                if str(cls_id) not in non_label_ids
            ]

    return md_results
