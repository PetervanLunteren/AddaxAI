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

from dataclasses import dataclass, field

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExclusionRollupResult:
    """Result from filter_and_rollup_classifications."""

    classifications: list[list]
    new_entries: list[dict] = field(default_factory=list)

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


def _build_taxonomy_key(
    entry: dict[str, str], target_level: str
) -> str:
    """
    Build a geofence-format taxonomy key for an ancestor level.

    Format: class;order;family;genus;species
    Fills levels up to target_level from the taxonomy entry,
    leaves more specific levels empty.

    Example: entry={"class":"mammalia","order":"artiodactyla",...},
    target_level="order" → "mammalia;artiodactyla;;;"
    """
    from app.ml.taxonomic_rollup import TAXONOMY_LEVELS

    parts = []
    for level in TAXONOMY_LEVELS:
        if level in entry and TAXONOMY_LEVELS.index(level) <= TAXONOMY_LEVELS.index(target_level):
            parts.append(entry[level])
        else:
            parts.append("")
    return ";".join(parts)


def filter_and_rollup_classifications(
    classifications: list[list],
    excluded_class_ids: set[str],
    class_id_to_name: dict[str, str],
    taxonomy_lookup: dict[str, dict[str, str]],
    classification_categories: dict[str, str],
    allowed_taxonomy_keys: frozenset[str] | None = None,
) -> ExclusionRollupResult:
    """
    Filter excluded labels and redirect their scores to taxonomy ancestors.

    Instead of removing excluded species and renormalizing (which inflates
    garbage predictions), redirects each excluded species' confidence to
    the nearest allowed ancestor in the taxonomy tree.

    Example: giraffe (90%) excluded in NLD → artiodactyla order

    When allowed_taxonomy_keys is provided (from geofence), the rollup
    walks up until it finds an ancestor whose taxonomy key is in the
    allowed set. Without it (manual exclusion), walks up to the first
    level above the excluded species.

    Args:
        classifications: List of [class_id, confidence] pairs
        excluded_class_ids: Set of class IDs to exclude
        class_id_to_name: Mapping of class_id -> class_name
        taxonomy_lookup: Mapping of model_class (lowercase) -> {level: taxon}
        classification_categories: Mutable dict of class_id -> name
            (new ancestor labels are added here in-memory)
        allowed_taxonomy_keys: Optional frozenset of taxonomy keys
            allowed by geofence (e.g., 'mammalia;artiodactyla;;;')

    Returns:
        New classification list with excluded scores redirected to
        ancestors, sorted by confidence descending.
    """
    if not classifications or not excluded_class_ids:
        return ExclusionRollupResult(classifications=classifications)

    from app.ml.taxonomic_rollup import TAXONOMY_LEVELS, _format_rollup_label

    # Build name-based excluded set for ancestor checking
    excluded_names = {
        class_id_to_name.get(str(cid), "").lower()
        for cid in excluded_class_ids
    }

    # Build reverse lookup: name (lowercase) -> class_id
    name_to_id: dict[str, str] = {}
    for cid, name in classification_categories.items():
        name_to_id[name.lower()] = cid

    # Track next available class_id for new ancestor labels
    max_id = max(
        (int(k) for k in classification_categories if k.isdigit()),
        default=0,
    )

    # Accumulate: kept items stay, excluded items redirect to ancestors
    kept: dict[str, float] = {}  # class_id -> confidence
    ancestor_scores: dict[str, float] = {}  # ancestor_label -> accumulated confidence
    ancestor_levels: dict[str, str] = {}  # ancestor_label -> taxonomy level

    for cls_id, conf in classifications:
        cls_id_str = str(cls_id)

        if cls_id_str not in excluded_class_ids:
            # Not excluded: keep as-is
            kept[cls_id_str] = kept.get(cls_id_str, 0.0) + conf
            continue

        # Excluded: find nearest non-excluded ancestor
        name = class_id_to_name.get(cls_id_str, "").lower()
        entry = taxonomy_lookup.get(name)

        if not entry:
            # No taxonomy info: drop the score (can't roll up)
            continue

        # Walk up taxonomy levels to find nearest ALLOWED ancestor.
        # Start from the level ABOVE the excluded class's most specific
        # level, since the class itself is what we're trying to replace.
        most_specific = None
        for level in reversed(TAXONOMY_LEVELS):
            if level in entry:
                most_specific = level
                break

        ancestor_label = None
        ancestor_level = None
        broadest_label = None
        broadest_level = None
        started = False

        for level in reversed(TAXONOMY_LEVELS):
            if level not in entry:
                continue

            # Skip the excluded class's own level
            if level == most_specific:
                started = True
                continue
            if not started:
                continue

            taxon_value = entry[level]
            label = _format_rollup_label(
                level, taxon_value, taxonomy_lookup
            )

            # Track broadest available as fallback
            if broadest_label is None:
                broadest_label = label
                broadest_level = level

            # Check if this ancestor is allowed
            if allowed_taxonomy_keys is not None:
                # Geofence mode: build the taxonomy key for this
                # ancestor and check if it's in the allowed set
                ancestor_key = _build_taxonomy_key(entry, level)
                if ancestor_key in allowed_taxonomy_keys:
                    ancestor_label = label
                    ancestor_level = level
                    break
            else:
                # Manual exclusion: accept first ancestor above
                # the excluded species (no geofence to check)
                if label.lower() not in excluded_names:
                    ancestor_label = label
                    ancestor_level = level
                    break

        # If no allowed ancestor, use broadest available
        if ancestor_label is None:
            ancestor_label = broadest_label
            ancestor_level = broadest_level

        if ancestor_label is None:
            # No taxonomy levels at all: drop the score
            continue

        ancestor_scores[ancestor_label] = (
            ancestor_scores.get(ancestor_label, 0.0) + conf
        )
        if ancestor_label not in ancestor_levels and ancestor_level:
            ancestor_levels[ancestor_label] = ancestor_level

    # Ensure ancestor labels have class IDs in classification_categories
    new_entries: list[dict] = []
    for label in ancestor_scores:
        if label.lower() not in name_to_id:
            max_id += 1
            new_id = str(max_id)
            classification_categories[new_id] = label
            name_to_id[label.lower()] = new_id
            new_entries.append({
                "name": label.lower(),
                "level": ancestor_levels.get(label, "unknown"),
            })

    # Build final classification list
    result: list[list] = []

    for cls_id, conf in kept.items():
        result.append([cls_id, conf])

    for label, conf in ancestor_scores.items():
        cls_id = name_to_id[label.lower()]
        # Merge with any existing kept score for this ancestor
        existing = next((r for r in result if r[0] == cls_id), None)
        if existing:
            existing[1] += conf
        else:
            result.append([cls_id, conf])

    result.sort(key=lambda x: x[1], reverse=True)
    return ExclusionRollupResult(
        classifications=result, new_entries=new_entries
    )


def should_skip_detection(
    det: dict,
    user_excluded_class_ids: set[str],
    non_label_class_ids: set[str],
    class_id_to_name: dict[str, str] | None = None,
    taxonomy_lookup: dict[str, dict[str, str]] | None = None,
    classification_categories: dict[str, str] | None = None,
    allowed_taxonomy_keys: frozenset[str] | None = None,
) -> bool:
    """
    Return True if a detection should not be loaded to the database.

    Steps:
    1. If no classifications, don't skip (unclassified animal).
    2. Apply user exclusion (with rollup if taxonomy available).
    3. If nothing remains after filtering, skip.
    4. If filtered top-1 is a NON_LABEL class, skip (false positive bbox).

    Args:
        det: Detection dict from JSON
        user_excluded_class_ids: Class IDs from user's excluded_classes
        non_label_class_ids: Class IDs for NON_LABEL_CLASSES
        class_id_to_name: Optional mapping for rollup support
        taxonomy_lookup: Optional taxonomy for rollup support
        classification_categories: Optional mutable dict for rollup support

    Returns:
        True if detection should be skipped, False if it should be loaded.
    """
    raw = det.get("classifications")
    if not raw:
        return False

    # Apply user exclusions (with rollup if taxonomy available)
    if user_excluded_class_ids:
        if taxonomy_lookup and class_id_to_name and classification_categories:
            filtered = filter_and_rollup_classifications(
                raw,
                user_excluded_class_ids,
                class_id_to_name,
                taxonomy_lookup,
                classification_categories,
                allowed_taxonomy_keys,
            ).classifications
        else:
            filtered = filter_classifications(raw, user_excluded_class_ids)
    else:
        filtered = raw

    # Nothing left after filtering
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
    taxonomy_lookup: dict[str, dict[str, str]] | None = None,
) -> dict:
    """
    Apply label exclusion to a full MegaDetector JSON results dict (in place).

    Always excludes NON_LABEL_CLASSES (bait, blank, empty, false detection,
    none, vide). Additionally excludes any user-configured labels.

    When taxonomy_lookup is provided, user-excluded species redirect their
    scores to taxonomy ancestors instead of being removed.

    Args:
        md_results: Full MegaDetector JSON dict (modified in place)
        excluded_labels: Optional list of label names to exclude
        taxonomy_lookup: Optional taxonomy for exclusion rollup

    Returns:
        The modified dict (same reference as input)
    """
    class_categories = md_results.get("classification_categories", {})
    excluded_class_ids = build_excluded_class_ids(class_categories, excluded_labels)

    if not excluded_class_ids:
        return md_results

    # Split into user exclusions and non-label IDs
    user_excluded_ids = build_user_excluded_class_ids(
        class_categories, excluded_labels
    )
    class_id_to_name = {str(k): v for k, v in class_categories.items()}

    use_rollup = bool(taxonomy_lookup and user_excluded_ids)

    for img in md_results.get("images", []):
        for det in img.get("detections", []):
            if "classifications" not in det or not det["classifications"]:
                continue

            if use_rollup:
                # Rollup user-excluded scores to ancestors, then strip NON_LABEL
                rollup_result = filter_and_rollup_classifications(
                    det["classifications"],
                    user_excluded_ids,
                    class_id_to_name,
                    taxonomy_lookup,
                    class_categories,
                )
                # Also strip NON_LABEL classes (for smoothing/rollup)
                non_label_ids = build_non_label_class_ids(class_categories)
                det["classifications"] = filter_classifications(
                    rollup_result.classifications, non_label_ids
                )
            else:
                det["classifications"] = filter_classifications(
                    det["classifications"], excluded_class_ids
                )

    return md_results
