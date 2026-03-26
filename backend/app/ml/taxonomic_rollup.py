"""
Taxonomic rollup: sum class probabilities at each taxonomic level
and pick the most specific level crossing the confidence threshold.

If the model's confidence at the species level is below the threshold,
rolls up to the next higher taxonomic level at which the summed
confidence reaches the threshold.

Runs BEFORE event smoothing so the smoother can refine rolled-up labels
using image/sequence context (e.g. "felidae" → "lion" if nearby
detections are confidently "lion").
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging_config import get_logger

logger = get_logger(__name__)

TAXONOMY_LEVELS = ["class", "order", "family", "genus", "species"]  # broadest → most specific
ROLLUP_THRESHOLD = 0.65


def format_latin_display_name(
    label: str,
    taxonomy_lookup: dict[str, dict[str, str]],
) -> str:
    """
    Format a label as its Latin taxonomy name for UI display.

    Rules:
        Species: 'G. camelopardalis' (abbreviated genus + epithet)
        Genus: 'Giraffa'
        Family: 'Felidae'
        Order: 'Artiodactyla'
        Class: 'Mammalia'
        No taxonomy: return label with first letter capitalized

    Args:
        label: Raw model class name (e.g., 'giraffe', 'domestic cattle')
        taxonomy_lookup: Mapping of model_class (lowercase) -> {level: taxon}

    Returns:
        Formatted Latin display name
    """
    if not label:
        return label

    entry = taxonomy_lookup.get(label.lower())
    if not entry:
        return label[0].upper() + label[1:] if label else label

    if "species" in entry and "genus" in entry:
        genus = entry["genus"]
        species = entry["species"]
        return f"{genus[0].upper()}. {species}"
    if "genus" in entry:
        return entry["genus"].capitalize()
    if "family" in entry:
        return entry["family"].capitalize()
    if "order" in entry:
        return entry["order"].capitalize()
    if "class" in entry:
        return entry["class"].capitalize()

    return label[0].upper() + label[1:] if label else label


@dataclass
class RollupResult:
    """Result of applying taxonomic rollup to MegaDetector JSON."""

    md_results: dict
    new_entries: list[dict] = field(default_factory=list)


def load_taxonomy_lookup(csv_path: Path) -> dict[str, dict[str, str]]:
    """
    Load taxonomy.csv into a lookup keyed by model_class (lowercased).

    Returns:
        Dict mapping model_class -> {"class": "mammalia", "order": "carnivora", ...}
        Only includes non-empty taxon values.

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Taxonomy CSV not found: {csv_path}")

    lookup: dict[str, dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_class = row.get("model_class", "").strip().lower()
            if not model_class:
                continue
            taxon = {}
            for level in TAXONOMY_LEVELS:
                val = row.get(level, "").strip().lower()
                if val:
                    taxon[level] = val
            if taxon:
                lookup[model_class] = taxon

    return lookup


def _format_rollup_label(
    level: str, taxon_value: str, taxonomy_lookup: dict[str, dict[str, str]],
) -> str:
    """Format the display label for a rolled-up detection."""
    if level == "species":
        # Find genus from any taxonomy entry with that species value
        for entry in taxonomy_lookup.values():
            if entry.get("species") == taxon_value and "genus" in entry:
                return f"{entry['genus']} {taxon_value}"
        return taxon_value
    return taxon_value


def _build_rollup_description(
    level: str, taxon_value: str, taxonomy_lookup: dict[str, dict[str, str]],
) -> str:
    """
    Build a 7-token classification_category_description for a rolled-up label.

    Format: name;class;order;family;genus;species;name
    Fills in ancestor levels from any taxonomy entry that has this taxon value.
    """
    # Find a taxonomy entry with this taxon to get ancestor levels
    ancestors: dict[str, str] = {}
    for entry in taxonomy_lookup.values():
        if entry.get(level) == taxon_value:
            ancestors = entry
            break

    label = _format_rollup_label(level, taxon_value, taxonomy_lookup)
    tokens = [label]
    for lvl in TAXONOMY_LEVELS:
        if lvl == level:
            tokens.append(taxon_value)
        elif TAXONOMY_LEVELS.index(lvl) < TAXONOMY_LEVELS.index(level):
            # Ancestor level — fill from taxonomy entry
            tokens.append(ancestors.get(lvl, ""))
        else:
            # More specific than rollup level — leave empty
            tokens.append("")
    tokens.append(label)
    return ";".join(tokens)


def rollup_single_detection(
    classifications: list[list],
    class_id_to_name: dict[str, str],
    taxonomy_lookup: dict[str, dict[str, str]],
) -> dict | None:
    """
    Apply taxonomic rollup to a single detection's classifications.

    Args:
        classifications: List of [class_id, confidence] pairs (sorted by conf desc)
        class_id_to_name: Mapping of class_id -> class_name
        taxonomy_lookup: Mapping of model_class -> {level: taxon_value}

    Returns:
        None if detection should be skipped (already confident or non-taxonomic),
        or {"label": str, "confidence": float, "level": str} if rolled up.
    """
    if not classifications:
        return None

    top_id, top_conf = classifications[0]
    top_name = class_id_to_name.get(str(top_id), "").lower()

    # Short-circuit: top-1 already confident enough
    if top_conf >= ROLLUP_THRESHOLD:
        return None

    # Skip non-taxonomic classes (not in taxonomy CSV).
    # Non-label classes (blank, empty, false detection, none) are already
    # stripped by label exclusion before rollup runs.
    if top_name not in taxonomy_lookup:
        return None

    # Sum confidences into level_sums[level][taxon_value]
    level_sums: dict[str, dict[str, float]] = {level: {} for level in TAXONOMY_LEVELS}

    for cls_id, conf in classifications:
        name = class_id_to_name.get(str(cls_id), "").lower()
        if name not in taxonomy_lookup:
            continue
        entry = taxonomy_lookup[name]
        for level in TAXONOMY_LEVELS:
            if level in entry:
                taxon = entry[level]
                level_sums[level][taxon] = level_sums[level].get(taxon, 0.0) + conf

    # Walk from most specific (species) to broadest (class)
    for level in reversed(TAXONOMY_LEVELS):
        sums = level_sums[level]
        if not sums:
            continue
        top_taxon = max(sums, key=sums.get)
        if sums[top_taxon] >= ROLLUP_THRESHOLD:
            label = _format_rollup_label(level, top_taxon, taxonomy_lookup)
            return {
                "label": label, "confidence": sums[top_taxon],
                "level": level, "taxon": top_taxon,
            }

    # Fallback: walk broadest to most specific, return first available
    for level in TAXONOMY_LEVELS:
        sums = level_sums[level]
        if sums:
            top_taxon = max(sums, key=sums.get)
            label = _format_rollup_label(level, top_taxon, taxonomy_lookup)
            return {
                "label": label, "confidence": sums[top_taxon],
                "level": level, "taxon": top_taxon,
            }

    return None


def apply_taxonomic_rollup_to_results(md_results: dict, taxonomy_csv_path: Path) -> RollupResult:
    """
    Apply taxonomic rollup to all detections in a MegaDetector JSON dict (in place).

    For each detection whose top-1 confidence is below the threshold and whose
    top-1 class is in the taxonomy, sums probabilities at each taxonomic level
    and picks the most specific level crossing the threshold.

    Args:
        md_results: Full MegaDetector JSON dict (modified in place)
        taxonomy_csv_path: Path to taxonomy.csv

    Returns:
        RollupResult with the modified dict and list of new rolled-up entries.
    """
    taxonomy_lookup = load_taxonomy_lookup(taxonomy_csv_path)
    if not taxonomy_lookup:
        return RollupResult(md_results=md_results)

    class_cats = md_results.get("classification_categories", {})
    if not class_cats:
        return RollupResult(md_results=md_results)

    # Build reverse mapping: class_id -> class_name
    class_id_to_name = {str(k): v for k, v in class_cats.items()}

    # Track new labels that need category IDs
    existing_names = {v.lower(): k for k, v in class_cats.items()}
    max_id = max((int(k) for k in class_cats if k.isdigit()), default=0)
    descriptions = md_results.setdefault("classification_category_descriptions", {})

    rolled_up = 0
    skipped_confident = 0
    skipped_non_taxonomic = 0
    new_entries: list[dict] = []
    seen_rollup_labels: set[str] = set()

    for img in md_results.get("images", []):
        for det in img.get("detections", []):
            classifications = det.get("classifications")
            if not classifications:
                continue

            result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
            if result is None:
                # Determine skip reason for logging
                top_conf = classifications[0][1]
                top_name = class_id_to_name.get(str(classifications[0][0]), "").lower()
                if top_conf >= ROLLUP_THRESHOLD:
                    skipped_confident += 1
                elif top_name not in taxonomy_lookup:
                    skipped_non_taxonomic += 1
                continue

            # Find or create category ID for the rolled-up label
            label = result["label"]
            if label.lower() in existing_names:
                new_id = existing_names[label.lower()]
            else:
                max_id += 1
                new_id = str(max_id)
                class_cats[new_id] = label
                class_id_to_name[new_id] = label
                existing_names[label.lower()] = new_id
                # Add description so MegaDetector's smoothing can work with this category
                descriptions[new_id] = _build_rollup_description(
                    result["level"], result["taxon"], taxonomy_lookup,
                )

            # Track new rolled-up entries (deduplicate by label)
            if label.lower() not in seen_rollup_labels:
                seen_rollup_labels.add(label.lower())
                new_entries.append({
                    "name": label.lower(),
                    "level": result["level"],
                })

            det["classifications"] = [[new_id, round(result["confidence"], 5)]]
            rolled_up += 1

    logger.info(
        f"Taxonomic rollup: {rolled_up} rolled up, "
        f"{skipped_confident} skipped (confident), "
        f"{skipped_non_taxonomic} skipped (non-taxonomic)"
    )

    return RollupResult(md_results=md_results, new_entries=new_entries)
