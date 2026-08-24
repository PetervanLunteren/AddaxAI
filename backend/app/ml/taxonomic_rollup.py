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

# ROLLUP_THRESHOLD is fixed policy (not per-project); it lives in
# app.core.confidence with the rest of the confidence defaults and is
# re-exported here because every rollup function defaults to it.
from app.core.confidence import ROLLUP_THRESHOLD
from app.core.logging_config import get_logger

logger = get_logger(__name__)

TAXONOMY_LEVELS = ["class", "order", "family", "genus", "species"]  # broadest → most specific


def format_scientific_name_from_taxonomy_row(
    label: str,
    taxon_genus: str | None,
    taxon_species: str | None,
    taxon_family: str | None = None,
    taxon_order: str | None = None,
    taxon_class: str | None = None,
    taxon_variant: str | None = None,
) -> str:
    """
    Format a Latin display name from individual taxonomy fields.

    Useful when you have a LabelTaxonomy row or individual fields
    rather than a full taxonomy_lookup dict.

    A variant is appended in parentheses ("V. vulpes (adult)") so two
    variant classes of one species never share a scientific name; the
    parentheses make clear it is not part of the Latin name.
    """
    if taxon_species and taxon_genus:
        base = f"{taxon_genus[0].upper()}. {taxon_species}"
    elif taxon_genus:
        base = taxon_genus.capitalize()
    elif taxon_family:
        base = taxon_family.capitalize()
    elif taxon_order:
        base = taxon_order.capitalize()
    elif taxon_class:
        base = taxon_class.capitalize()
    else:
        base = label[0].upper() + label[1:] if label else label
    if taxon_variant:
        return f"{base} ({taxon_variant})"
    return base


def format_common_name(label: str) -> str:
    """
    Clean a class label into a common name: underscores to spaces and
    capitalise the first letter. Mirrors the frontend ``normalizeLabel``
    so common-mode names match across the UI and stored data.

    ``label`` already degrades to the Latin taxon where SpeciesNet had no
    common name (e.g. rollups), so this also yields the right value there.
    """
    if not label:
        return label
    cleaned = label.replace("_", " ")
    return cleaned[0].upper() + cleaned[1:]


def format_leaf_annotation(label: str, scientific_name: str, level: str) -> str:
    """
    The qualifier rendered in italics beside a taxonomy tree leaf.

    Both trees show the scientific name as the leaf's name, so the qualifier
    carries the model's own label for that taxon, which is the second and
    usually more recognisable name: ``Papio (baboon)``. When the label
    already *is* the scientific name there is no second name to give, so the
    qualifier names the rank instead: ``Gorilla (genus)``.

    Shared by the model taxonomy tree (``ml.taxonomy_parser``, built from a
    model's taxonomy.csv) and the label filter tree
    (``api.crud.label_tree``, built from the label_taxonomy table) so the
    same taxon reads identically in the species picker and in the Labels
    filter. Keep it that way: the two trees are rendered by one component,
    so a divergence here shows up as two rows that look alike and mean
    different things.
    """
    cleaned = label.replace("_", " ")
    if cleaned.lower() != scientific_name.lower():
        return cleaned
    return level


def resolve_label_names(
    label: str | None,
    taxonomy: object | None,
    category: str,
) -> tuple[str | None, str | None]:
    """
    Single source of truth for a detection's two display names.

    Returns ``(common_name, scientific_name)``:
    - With a LabelTaxonomy row, copy its precomputed names (the row is
      where the formatting actually happens, in ``taxonomy_db``).
    - With a label but no taxonomy row (e.g. human relabel to a custom
      label), clean the label for common and capitalise it for scientific.
    - Unclassified (no label): both fall back to the capitalised category.

    Never reads model output; safe to run over verified detections because
    it only derives display strings from the existing label / taxonomy.
    """
    if taxonomy is not None and (
        getattr(taxonomy, "scientific_name", None)
        or getattr(taxonomy, "common_name", None)
    ):
        return taxonomy.common_name, taxonomy.scientific_name
    if label:
        return format_common_name(label), label[0].upper() + label[1:]
    cap = category.capitalize() if category else category
    return cap, cap


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


def _format_rollup_label(level: str, entry: dict[str, str]) -> str:
    """Format the display label for a rolled-up detection.

    ``entry`` is the representative taxonomy entry the rollup summed on,
    so the genus is read from the same chain the taxon came from, never
    looked up by value (two genera can share a species epithet).
    """
    taxon_value = entry[level]
    if level == "species" and "genus" in entry:
        return f"{entry['genus']} {taxon_value}"
    return taxon_value


def _build_rollup_description(
    level: str, taxon_value: str, ancestors: dict[str, str],
) -> str:
    """
    Build a 7-token classification_category_description for a rolled-up label.

    Format: name;class;order;family;genus;species;name
    ``ancestors`` is the representative taxonomy entry the rollup summed
    on (empty for the kingdom fallback, which has no taxonomy fields).
    """
    if level == "kingdom":
        return f"{taxon_value};;;;;;{taxon_value}"

    label = _format_rollup_label(level, ancestors)
    tokens = [label]
    for lvl in TAXONOMY_LEVELS:
        if lvl == level:
            tokens.append(taxon_value)
        elif TAXONOMY_LEVELS.index(lvl) < TAXONOMY_LEVELS.index(level):
            # Ancestor level — fill from the representative entry
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
    threshold: float = ROLLUP_THRESHOLD,
) -> dict | None:
    """
    Apply taxonomic rollup to a single detection's classifications.

    Two rollup paths (matching the official SpeciesNet API):

    **Path A (geofence rollup)**: top-1 is excluded. Roll up to the
    nearest allowed ancestor using top-1 confidence as the threshold.
    Walks family, order, class (skips species and genus).

    **Path B (confidence rollup)**: top-1 is allowed but confidence
    < ``threshold``. Roll up to the nearest level above ``threshold``.
    Walks genus, family, order, class.

    Both paths use only the top-5 predictions for summing (matching
    the official SpeciesNet classifier which returns top-5 only).
    The rolled-up result must pass the geofence check
    (allowed_taxonomy_keys).

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
        threshold: Confidence floor for Path B and for the kingdom-level
            fallback. Defaults to the fixed ``ROLLUP_THRESHOLD`` (0.65,
            in app.core.confidence); the parameter stays overridable so
            tests can exercise other values.

    Returns:
        None if no rollup found (keep raw top-1 as-is).
        {"label": str, "confidence": float, "level": str, "taxon": str}
            if rollup found a confident ancestor.
    """
    if not classifications:
        return None

    top_id, top_conf = classifications[0]
    top_name = class_id_to_name.get(str(top_id), "").lower()

    # Non-taxonomic top-1 (blank, vehicle, etc.) - run rollup to see
    # if the other top-5 entries can roll up to "animal" at kingdom
    # level. If not, keep the raw top-1 (matches official API wrapper
    # which keeps raw top-1 when rollup returns None).
    top_in_taxonomy = top_name in taxonomy_lookup

    # Determine rollup path
    top_is_excluded = (
        excluded_names is not None and top_name in excluded_names
    )

    top_is_species = (
        top_in_taxonomy
        and "species" in taxonomy_lookup.get(top_name, {})
    )
    if (
        top_in_taxonomy
        and not top_is_excluded
        and top_conf >= threshold
        and top_is_species
    ):
        # Top-1 is a confident species-level prediction: no rollup needed.
        # Non-species labels (e.g., "bird", "bovidae family") always go
        # through rollup to sum the top-5 for a more accurate confidence.
        return None

    # Use top-5 predictions for sums (matches official SpeciesNet API
    # which only returns top-5 from its classifier)
    top5 = classifications[:5]

    # Sum top-5 scores at each taxonomy level, keyed by the full
    # ancestor-chain key ("class;order;family;genus;species" trimmed to
    # the level), never by the bare taxon value: species epithets repeat
    # across genera (four "canadensis" classes in one real model), and a
    # bare-value key summed them together and labelled the result from
    # whichever entry matched first. Also track a representative entry
    # per key for the allowed check and label formatting.
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
                key = _build_taxonomy_key_for_level(entry, level)
                level_sums[level][key] = (
                    level_sums[level].get(key, 0.0) + conf
                )
                if key not in level_entries[level]:
                    level_entries[level][key] = entry

    if top_is_excluded:
        # Path A: geofence rollup
        walk_threshold = top_conf
        walk_levels = ["family", "order", "class"]
    else:
        # Path B: confidence rollup. Species sits at the front so models
        # with multiple classes per species (e.g. age or sex variants of
        # one species) can roll up sibling probabilities to the shared
        # species before falling back to genus.
        walk_threshold = threshold
        walk_levels = ["species", "genus", "family", "order", "class"]

    # Walk from most specific to broadest. At each level, find the
    # max-scoring taxon that crosses the threshold and is allowed
    # (matching official SpeciesNet geofence_utils.py lines 186-196).
    for level in walk_levels:
        sums = level_sums.get(level, {})
        if not sums:
            continue
        for key in sorted(sums, key=sums.get, reverse=True):
            if sums[key] < walk_threshold:
                break  # remaining taxa have even lower scores
            if (
                allowed_taxonomy_keys is not None
                and key not in allowed_taxonomy_keys
            ):
                continue  # geofence: ancestor not allowed
            entry = level_entries[level][key]
            return {
                "label": _format_rollup_label(level, entry),
                "confidence": sums[key],
                "level": level,
                "taxon": entry[level],
                "ancestors": entry,
            }

    # No level crossed the threshold with an allowed result.
    # Try kingdom level (last resort): sum all top-5 with any taxonomy
    # info. If the sum >= threshold, return "animal" at that score.
    # Matches official API behavior of rolling up to kingdom.
    kingdom_sum = sum(
        conf for cls_id, conf in top5
        if class_id_to_name.get(str(cls_id), "").lower() in taxonomy_lookup
    )
    if kingdom_sum >= threshold:
        return {
            "label": "animal",
            "confidence": kingdom_sum,
            "level": "kingdom",
            "taxon": "animal",
            "ancestors": {},
        }

    # Nothing crossed any threshold. Return None to keep the raw top-1
    # (matches official API wrapper run_md_and_speciesnet behavior).
    return None


def apply_taxonomic_rollup_to_results(
    md_results: dict,
    taxonomy_csv_path: Path,
    excluded_names: frozenset[str] | None = None,
    allowed_taxonomy_keys: frozenset[str] | None = None,
    threshold: float = ROLLUP_THRESHOLD,
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
        threshold: Confidence floor for Path B and kingdom fallback.
            Defaults to the fixed ``ROLLUP_THRESHOLD`` (0.65, in
            app.core.confidence); overridable for tests.

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

    rolled_up = 0
    skipped = 0
    new_entries: list[dict] = []
    seen_rollup_labels: set[str] = set()

    # `images or []` / `detections or []` keeps the rollup safe against
    # process_video failure entries (`detections: null` for corrupt
    # videos). They contribute zero classifications either way.
    for img in md_results.get("images") or []:
        for det in img.get("detections") or []:
            classifications = det.get("classifications")
            if not classifications:
                continue

            result = rollup_single_detection(
                classifications,
                class_id_to_name,
                taxonomy_lookup,
                excluded_names=excluded_names,
                allowed_taxonomy_keys=allowed_taxonomy_keys,
                threshold=threshold,
            )
            if result is None:
                # No rollup needed or rollup found nothing - keep raw top-1
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
                    result["level"], result["taxon"], result["ancestors"],
                )

            # Track new rolled-up entries (deduplicate by label). The
            # ancestors ride along so the DB entry never has to find
            # them again by value search.
            if label.lower() not in seen_rollup_labels:
                seen_rollup_labels.add(label.lower())
                new_entries.append({
                    "name": label.lower(),
                    "level": result["level"],
                    "ancestors": result["ancestors"],
                })

            det["classifications"] = [
                [new_id, round(result["confidence"], 5)]
            ]
            rolled_up += 1

    logger.info(
        f"Taxonomic rollup: {rolled_up} rolled up, {skipped} skipped"
    )

    return RollupResult(md_results=md_results, new_entries=new_entries)
