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


def format_display_name_from_taxonomy_row(
    label: str,
    taxon_genus: str | None,
    taxon_species: str | None,
    taxon_family: str | None = None,
    taxon_order: str | None = None,
    taxon_class: str | None = None,
) -> str:
    """
    Format a Latin display name from individual taxonomy fields.

    Useful when you have a LabelTaxonomy row or individual fields
    rather than a full taxonomy_lookup dict.
    """
    if taxon_species and taxon_genus:
        return f"{taxon_genus[0].upper()}. {taxon_species}"
    if taxon_genus:
        return taxon_genus.capitalize()
    if taxon_family:
        return taxon_family.capitalize()
    if taxon_order:
        return taxon_order.capitalize()
    if taxon_class:
        return taxon_class.capitalize()
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


def _build_taxonomy_key_for_level(
    entry: dict[str, str], target_level: str,
) -> str:
    """
    Build a geofence-format taxonomy key for a taxon at a given level.

    Format: class;order;family;genus;species
    Fills levels up to target_level from the taxonomy entry,
    leaves more specific levels empty.

    Duplicated from label_exclusion._build_taxonomy_key to avoid
    circular import (label_exclusion imports from this module).
    """
    parts: list[str] = []
    for level in TAXONOMY_LEVELS:
        if (
            level in entry
            and TAXONOMY_LEVELS.index(level)
            <= TAXONOMY_LEVELS.index(target_level)
        ):
            parts.append(entry[level])
        else:
            parts.append("")
    return ";".join(parts)


def rollup_single_detection(
    classifications: list[list],
    class_id_to_name: dict[str, str],
    taxonomy_lookup: dict[str, dict[str, str]],
    excluded_names: frozenset[str] | None = None,
    allowed_taxonomy_keys: frozenset[str] | None = None,
    included_ancestor_taxa: frozenset[tuple[str, str]] | None = None,
) -> dict | None:
    """
    Apply taxonomic rollup to a single detection's classifications.

    Two rollup paths (matching the official SpeciesNet API):

    **Path A (geofence rollup)**: top-1 is excluded. Roll up to the
    nearest allowed ancestor using top-1 confidence as the threshold.
    Walks family, order, class (skips species and genus).

    **Path B (confidence rollup)**: top-1 is allowed but confidence
    < 0.65. Roll up to the nearest level above 0.65. Walks genus,
    family, order, class.

    Both paths use only the top-5 predictions for summing (matching
    the official SpeciesNet classifier which returns top-5 only).
    The rolled-up result must pass both the geofence check
    (allowed_taxonomy_keys) and the user exclusion check
    (included_ancestor_taxa).

    Args:
        classifications: [class_id, confidence] pairs (sorted by conf desc)
        class_id_to_name: Mapping of class_id -> class_name
        taxonomy_lookup: Mapping of model_class -> {level: taxon_value}
        excluded_names: Lowercase names of excluded species. When provided,
            enables Path A (geofence rollup) for excluded top-1 species.
        allowed_taxonomy_keys: Geofence taxonomy keys
            (format "class;order;family;genus;species") that are allowed
            in the project's country. Rollup candidates are checked
            against this set.
        included_ancestor_taxa: Pre-computed set of (level, taxon)
            pairs that have at least one non-excluded descendant.
            Rollup candidates without included descendants are skipped.

    Returns:
        None if no rollup needed or nothing qualifies,
        or {"label": str, "confidence": float, "level": str, "taxon": str}.
    """
    if not classifications:
        return None

    top_id, top_conf = classifications[0]
    top_name = class_id_to_name.get(str(top_id), "").lower()

    # Skip non-taxonomic classes (blank, vehicle, human, etc.)
    if top_name not in taxonomy_lookup:
        return None

    # Determine rollup path
    top_is_excluded = (
        excluded_names is not None and top_name in excluded_names
    )

    top_is_species = "species" in taxonomy_lookup.get(top_name, {})
    if not top_is_excluded and top_conf >= ROLLUP_THRESHOLD and top_is_species:
        # Top-1 is a confident species-level prediction: no rollup needed.
        # Non-species labels (e.g., "bird", "bovidae family") always go
        # through rollup to sum the top-5 for a more accurate confidence.
        return None

    # Use top-5 predictions for sums (matches official SpeciesNet API
    # which only returns top-5 from its classifier)
    top5 = classifications[:5]

    # Sum top-5 scores at each taxonomy level.
    # Also track a representative entry per (level, taxon) for the
    # allowed check.
    level_sums: dict[str, dict[str, float]] = {
        level: {} for level in TAXONOMY_LEVELS
    }
    level_entries: dict[str, dict[str, dict[str, str]]] = {
        level: {} for level in TAXONOMY_LEVELS
    }

    for cls_id, conf in top5:
        name = class_id_to_name.get(str(cls_id), "").lower()
        if name not in taxonomy_lookup:
            continue
        entry = taxonomy_lookup[name]
        for level in TAXONOMY_LEVELS:
            if level in entry:
                taxon = entry[level]
                level_sums[level][taxon] = (
                    level_sums[level].get(taxon, 0.0) + conf
                )
                if taxon not in level_entries[level]:
                    level_entries[level][taxon] = entry

    if top_is_excluded:
        # Path A: geofence rollup
        threshold = top_conf
        walk_levels = ["family", "order", "class"]
    else:
        # Path B: confidence rollup
        threshold = ROLLUP_THRESHOLD
        walk_levels = ["genus", "family", "order", "class"]

    # Walk from most specific to broadest. At each level, find the
    # max-scoring taxon that crosses the threshold and is allowed
    # (matching official SpeciesNet geofence_utils.py lines 186-196).
    for level in walk_levels:
        sums = level_sums.get(level, {})
        if not sums:
            continue
        for taxon in sorted(sums, key=sums.get, reverse=True):
            if sums[taxon] < threshold:
                break  # remaining taxa have even lower scores
            if allowed_taxonomy_keys is not None:
                entry = level_entries[level][taxon]
                key = _build_taxonomy_key_for_level(entry, level)
                if key not in allowed_taxonomy_keys:
                    continue  # geofence: ancestor not allowed
            if included_ancestor_taxa is not None:
                if (level, taxon) not in included_ancestor_taxa:
                    continue  # no included descendants
            label = _format_rollup_label(
                level, taxon, taxonomy_lookup
            )
            return {
                "label": label,
                "confidence": sums[taxon],
                "level": level,
                "taxon": taxon,
            }

    # No level crossed the threshold with an allowed result
    return None


def apply_taxonomic_rollup_to_results(
    md_results: dict,
    taxonomy_csv_path: Path,
    excluded_names: frozenset[str] | None = None,
    allowed_taxonomy_keys: frozenset[str] | None = None,
) -> RollupResult:
    """
    Apply taxonomic rollup to all detections in a MegaDetector JSON dict (in place).

    Supports two rollup paths via rollup_single_detection():
    - Path A (geofence rollup): top-1 is excluded, rolls up to allowed ancestor
    - Path B (confidence rollup): top-1 is allowed but low confidence

    Args:
        md_results: Full MegaDetector JSON dict (modified in place)
        taxonomy_csv_path: Path to taxonomy.csv
        excluded_names: Lowercase names of excluded species (enables Path A)
        allowed_taxonomy_keys: Geofence taxonomy keys allowed in the country

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
    descriptions = md_results.setdefault(
        "classification_category_descriptions", {}
    )

    # Pre-compute which (level, taxon) pairs have at least one
    # non-excluded descendant. Rollup candidates without included
    # descendants are skipped.
    included_ancestor_taxa: frozenset[tuple[str, str]] | None = None
    if excluded_names:
        _included: set[tuple[str, str]] = set()
        for model_class, entry in taxonomy_lookup.items():
            if model_class not in excluded_names:
                for level in TAXONOMY_LEVELS:
                    if level in entry:
                        _included.add((level, entry[level]))
        included_ancestor_taxa = frozenset(_included)

    rolled_up = 0
    skipped = 0
    new_entries: list[dict] = []
    seen_rollup_labels: set[str] = set()

    for img in md_results.get("images", []):
        for det in img.get("detections", []):
            classifications = det.get("classifications")
            if not classifications:
                continue

            result = rollup_single_detection(
                classifications,
                class_id_to_name,
                taxonomy_lookup,
                excluded_names=excluded_names,
                allowed_taxonomy_keys=allowed_taxonomy_keys,
                included_ancestor_taxa=included_ancestor_taxa,
            )
            if result is None:
                skipped += 1
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

            det["classifications"] = [
                [new_id, round(result["confidence"], 5)]
            ]
            rolled_up += 1

    logger.info(
        f"Taxonomic rollup: {rolled_up} rolled up, {skipped} skipped"
    )

    return RollupResult(md_results=md_results, new_entries=new_entries)
