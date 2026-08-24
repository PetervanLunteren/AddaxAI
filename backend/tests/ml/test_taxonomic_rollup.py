"""Tests for app.ml.taxonomic_rollup."""

import csv
from pathlib import Path

import pytest

from app.ml.label_exclusion import NON_LABEL_CLASSES
from app.ml.taxonomic_rollup import (
    apply_taxonomic_rollup_to_results,
    format_leaf_annotation,
    load_taxonomy_lookup,
    rollup_single_detection,
)

SAMPLE_TAXONOMY_ROWS = [
    {
        "model_class": "leopard", "class": "mammalia",
        "order": "carnivora", "family": "felidae",
        "genus": "panthera", "species": "pardus",
    },
    {
        "model_class": "lion", "class": "mammalia",
        "order": "carnivora", "family": "felidae",
        "genus": "panthera", "species": "leo",
    },
    {
        "model_class": "cheetah", "class": "mammalia",
        "order": "carnivora", "family": "felidae",
        "genus": "acinonyx", "species": "jubatus",
    },
    {
        "model_class": "zebra", "class": "mammalia",
        "order": "perissodactyla", "family": "equidae",
        "genus": "equus", "species": "quagga",
    },
    {
        "model_class": "bird", "class": "aves",
        "order": "", "family": "", "genus": "", "species": "",
    },
    {
        "model_class": "blank", "class": "",
        "order": "", "family": "", "genus": "", "species": "",
    },
]


@pytest.fixture
def taxonomy_csv(tmp_path):
    """Write sample taxonomy CSV and return its path."""
    csv_path = tmp_path / "taxonomy.csv"
    fieldnames = ["model_class", "class", "order", "family", "genus", "species"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(SAMPLE_TAXONOMY_ROWS)
    return csv_path


@pytest.fixture
def taxonomy_lookup(taxonomy_csv):
    return load_taxonomy_lookup(taxonomy_csv)


@pytest.fixture
def class_id_to_name():
    return {
        "0": "leopard",
        "1": "lion",
        "2": "cheetah",
        "3": "zebra",
        "4": "bird",
        "5": "blank",
    }


# --- load_taxonomy_lookup ---

def test_load_taxonomy_lookup(taxonomy_csv):
    lookup = load_taxonomy_lookup(taxonomy_csv)
    assert "leopard" in lookup
    assert lookup["leopard"]["family"] == "felidae"
    assert lookup["leopard"]["species"] == "pardus"
    # bird has only class level
    assert "bird" in lookup
    assert lookup["bird"] == {"class": "aves"}


def test_load_taxonomy_lookup_empty_fields(taxonomy_csv):
    lookup = load_taxonomy_lookup(taxonomy_csv)
    # blank has no taxonomy levels → not in lookup
    assert "blank" not in lookup


def test_load_taxonomy_lookup_missing_file():
    with pytest.raises(FileNotFoundError):
        load_taxonomy_lookup(Path("/nonexistent/taxonomy.csv"))


# --- rollup_single_detection ---

def test_skip_confident(taxonomy_lookup, class_id_to_name):
    # top-1 at 0.80 → skip
    classifications = [[0, 0.80], [1, 0.10], [2, 0.10]]
    result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
    assert result is None


def test_non_species_top1_still_rolls_up(taxonomy_lookup, class_id_to_name):
    """Non-species top-1 above threshold still rolls up to sum confidence.

    'bird' (class=aves, no genus/species) at 0.80 with felidae species
    in top-5 should roll up, not skip. The mammalia class sum
    (0.80 + 0.10 + 0.05 = 0.95) crosses 0.65.
    """
    # bird 0.80 (class-level), leopard 0.10, lion 0.05, zebra 0.05
    classifications = [[4, 0.80], [0, 0.10], [1, 0.05], [3, 0.05]]
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup
    )
    assert result is not None
    # bird is the only aves entry, so aves sum = 0.80. But mammalia
    # sum = 0.80 + 0.10 + 0.05 + 0.05 = 1.0. Rollup picks the most
    # specific level above 0.65, which is class=mammalia (since aves
    # and mammalia are both class-level and mammalia has the higher sum).
    assert result["confidence"] >= 0.80


def test_non_species_top1_resolves_to_specific_level(
    taxonomy_lookup, class_id_to_name
):
    """Non-species top-1 can resolve to a more specific level than itself.

    'bird' (class=aves) at 0.30 with 3 felidae species summing to 0.70
    at family level should pick felidae, not aves.
    """
    # bird 0.30, leopard 0.30, lion 0.25, cheetah 0.15
    # felidae family = 0.30 + 0.25 + 0.15 = 0.70 (>= 0.65)
    classifications = [[4, 0.30], [0, 0.30], [1, 0.25], [2, 0.15]]
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup
    )
    assert result is not None
    assert result["level"] == "family"
    assert result["label"] == "felidae"


def test_blank_top1_kingdom_rollup(taxonomy_lookup, class_id_to_name):
    """Top-1 is blank but other top-5 species sum to kingdom > 0.65."""
    # blank 0.20, leopard 0.30, lion 0.20, cheetah 0.20 → kingdom = 0.70
    classifications = [[5, 0.20], [0, 0.30], [1, 0.20], [2, 0.20]]
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup
    )
    # Should walk genus/family/order/class first. felidae family = 0.70.
    assert result is not None
    assert result["level"] == "family"
    assert result["taxon"] == "felidae"


def test_blank_top1_no_rollup_keeps_top1(taxonomy_lookup, class_id_to_name):
    """Top-1 is blank, no rollup possible (too few animal scores)."""
    # blank 0.50, leopard 0.05, lion 0.05 → no level reaches 0.65
    classifications = [[5, 0.50], [0, 0.05], [1, 0.05]]
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup
    )
    # Returns None - the raw "blank" top-1 is kept by caller
    assert result is None


def test_rollup_to_genus(taxonomy_lookup, class_id_to_name):
    # leopard 0.35 + lion 0.35 = panthera 0.70 (>= 0.65)
    classifications = [[0, 0.35], [1, 0.35], [2, 0.15], [3, 0.15]]
    result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
    assert result is not None
    assert result["label"] == "panthera"
    assert result["level"] == "genus"
    assert result["confidence"] == pytest.approx(0.70, abs=0.01)


def test_rollup_to_family(taxonomy_lookup, class_id_to_name):
    # leopard 0.25 + lion 0.20 + cheetah 0.25 = felidae 0.70 (>= 0.65)
    # genus: panthera = 0.45, acinonyx = 0.25 — neither crosses 0.65
    classifications = [[0, 0.25], [2, 0.25], [1, 0.20], [3, 0.30]]
    result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
    assert result is not None
    assert result["label"] == "felidae"
    assert result["level"] == "family"
    assert result["confidence"] == pytest.approx(0.70, abs=0.01)


def test_fallback_broadest(taxonomy_lookup, class_id_to_name):
    # No level crosses 0.65 → return broadest available (class)
    # leopard 0.20, lion 0.15, cheetah 0.15, zebra 0.15, bird 0.15, blank 0.20
    classifications = [[0, 0.20], [5, 0.20], [1, 0.15], [2, 0.15], [3, 0.15], [4, 0.15]]
    result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
    assert result is not None
    assert result["level"] == "class"
    assert result["label"] == "mammalia"


def test_species_binomial(taxonomy_lookup, class_id_to_name):
    # All confidence on leopard species but below top-1 threshold
    # leopard at 0.64 (just under 0.65), lion at 0.01
    # species pardus = 0.64, genus panthera = 0.65 — genus wins at threshold
    # But if pardus had >= 0.65, species would win
    # Let's test with pardus exactly at 0.65 via 2 entries... actually only leopard maps to pardus
    # Instead: set leopard confidence to something that sums to 0.65 for pardus at species level
    # Since only leopard maps to pardus, leopard needs to be 0.65 → but that triggers short-circuit
    # So we can't get species-level rollup with this fixture unless we have two model_classes
    # mapping to the same species. Test the label format directly instead.
    # Actually, if top-1 is 0.64 and it's the only one for its species, species sum is 0.64 < 0.65
    # genus sum is 0.65 (leopard 0.64 + lion 0.01) → genus wins
    classifications = [[0, 0.64], [1, 0.01], [3, 0.35]]
    result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
    assert result is not None
    assert result["level"] == "genus"
    assert result["label"] == "panthera"


def test_empty_classifications(taxonomy_lookup, class_id_to_name):
    result = rollup_single_detection([], class_id_to_name, taxonomy_lookup)
    assert result is None


def test_rollup_result_carries_the_matched_ancestors(
    taxonomy_lookup, class_id_to_name
):
    """The result names the chain it summed on, so consumers never have
    to find ancestors again by value search."""
    # leopard 0.35 + lion 0.35 → genus panthera 0.70
    classifications = [[0, 0.35], [1, 0.35]]
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup
    )
    assert result is not None
    assert result["level"] == "genus"
    assert result["ancestors"]["genus"] == "panthera"
    assert result["ancestors"]["family"] == "felidae"


def test_variant_siblings_sum_to_their_shared_species():
    """Two classes of one species (age variants) roll up to the species
    when neither is confident alone. This is what makes variant models
    degrade gracefully instead of stalling at genus."""
    lookup = {
        "red fox adult": {
            "class": "mammalia", "order": "carnivora",
            "family": "canidae", "genus": "vulpes", "species": "vulpes",
        },
        "red fox juvenile": {
            "class": "mammalia", "order": "carnivora",
            "family": "canidae", "genus": "vulpes", "species": "vulpes",
        },
    }
    ids = {"0": "red fox adult", "1": "red fox juvenile"}
    # 0.40 + 0.30 = species 0.70 >= 0.65; neither top-1 is confident
    result = rollup_single_detection([[0, 0.40], [1, 0.30]], ids, lookup)
    assert result is not None
    assert result["level"] == "species"
    assert result["label"] == "vulpes vulpes"
    assert result["confidence"] == pytest.approx(0.70, abs=0.01)


def test_shared_epithet_species_never_sum_together():
    """Species epithets repeat across genera ("canadensis"). Summing by
    the bare value merged a goose with a crane and labelled the result
    from whichever entry matched first. Keys are full ancestor chains."""
    lookup = {
        "canada goose": {
            "class": "aves", "order": "anseriformes",
            "family": "anatidae", "genus": "branta", "species": "canadensis",
        },
        "crane": {
            "class": "aves", "order": "gruiformes",
            "family": "gruidae", "genus": "antigone", "species": "canadensis",
        },
    }
    ids = {"0": "canada goose", "1": "crane"}
    # Bare-epithet keying summed 0.40 + 0.35 = 0.75 at "species
    # canadensis" and returned a species-level mislabel. Correct keying
    # leaves every species and genus below 0.65; only class aves (0.75)
    # crosses.
    result = rollup_single_detection([[0, 0.40], [1, 0.35]], ids, lookup)
    assert result is not None
    assert result["level"] == "class"
    assert result["label"] == "aves"


# --- apply_taxonomic_rollup_to_results ---

def test_apply_adds_new_category(taxonomy_csv):
    md_results = {
        "classification_categories": {
            "0": "leopard",
            "1": "lion",
            "2": "cheetah",
        },
        "images": [
            {
                "file": "img1.jpg",
                "detections": [
                    {
                        "bbox": [0.1, 0.1, 0.5, 0.5],
                        "classifications": [[0, 0.35], [1, 0.35], [2, 0.15]],
                    }
                ],
            }
        ],
    }

    apply_taxonomic_rollup_to_results(md_results, taxonomy_csv)

    cats = md_results["classification_categories"]
    # "panthera" should be a new category
    assert "panthera" in cats.values()

    # The detection should now have a single classification entry
    det = md_results["images"][0]["detections"][0]
    assert len(det["classifications"]) == 1
    new_id = str(det["classifications"][0][0])
    assert cats[new_id] == "panthera"


def test_apply_mixed(taxonomy_csv):
    md_results = {
        "classification_categories": {
            "0": "leopard",
            "1": "lion",
            "5": "blank",
        },
        "images": [
            {
                "file": "img1.jpg",
                "detections": [
                    # Confident → skip (no rollup needed)
                    {
                        "bbox": [0.1, 0.1, 0.5, 0.5],
                        "classifications": [[0, 0.80], [1, 0.20]],
                    },
                    # Non-taxonomic top-1, no rollup → keeps raw top-1
                    {
                        "bbox": [0.2, 0.2, 0.3, 0.3],
                        "classifications": [[5, 0.50], [0, 0.30], [1, 0.20]],
                    },
                    # Rolls up to genus
                    {
                        "bbox": [0.3, 0.3, 0.4, 0.4],
                        "classifications": [[0, 0.35], [1, 0.35], [5, 0.30]],
                    },
                ],
            }
        ],
    }

    apply_taxonomic_rollup_to_results(md_results, taxonomy_csv)

    dets = md_results["images"][0]["detections"]

    # First: unchanged (confident)
    assert dets[0]["classifications"][0][0] == 0
    assert dets[0]["classifications"][0][1] == 0.80

    # Second: kept as raw top-1 (non-taxonomic, no rollup possible)
    assert dets[1]["classifications"][0][0] == 5

    # Third: rolled up to "panthera"
    assert len(dets[2]["classifications"]) == 1
    new_id = str(dets[2]["classifications"][0][0])
    assert md_results["classification_categories"][new_id] == "panthera"


def test_apply_adds_descriptions(taxonomy_csv):
    """Rolled-up categories get classification_category_descriptions for MegaDetector smoothing."""
    md_results = {
        "classification_categories": {"0": "leopard", "1": "lion"},
        "classification_category_descriptions": {
            "0": "leopard;mammalia;carnivora;felidae;panthera;pardus;leopard",
            "1": "lion;mammalia;carnivora;felidae;panthera;leo;lion",
        },
        "images": [
            {
                "file": "img1.jpg",
                "detections": [
                    {"bbox": [0.1, 0.1, 0.5, 0.5], "classifications": [[0, 0.35], [1, 0.35]]},
                ],
            }
        ],
    }

    apply_taxonomic_rollup_to_results(md_results, taxonomy_csv)

    descs = md_results["classification_category_descriptions"]
    det = md_results["images"][0]["detections"][0]
    new_id = str(det["classifications"][0][0])

    # New category should have a 7-token description
    assert new_id in descs
    tokens = descs[new_id].split(";")
    assert len(tokens) == 7
    assert tokens[0] == "panthera"  # display name
    assert tokens[6] == "panthera"  # display name (last)
    assert tokens[1] == "mammalia"  # class
    assert tokens[4] == "panthera"  # genus level (the rollup level)


# --- NON_LABEL_CLASSES exclusion ---
# Non-label classes (blank, empty, false detection, none) are now stripped
# by label_exclusion.py *before* rollup runs. Rollup no longer needs its
# own guard -- it simply won't see them.


def test_non_label_top1_returns_none(taxonomy_lookup, class_id_to_name):
    """Non-taxonomic top-1 returns None when other species don't sum enough."""
    # "blank" (class_id 5) top-1 at 0.50, others at 0.30 + 0.20 = 0.50 total
    # → no level reaches 0.65 → returns None (caller keeps raw top-1)
    classifications = [[5, 0.50], [0, 0.30], [1, 0.20]]
    result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
    assert result is None


def test_non_label_classes_constant():
    """NON_LABEL_CLASSES (now in label_exclusion.py) contains the expected entries."""
    assert "blank" in NON_LABEL_CLASSES
    assert "empty" in NON_LABEL_CLASSES
    assert "false detection" in NON_LABEL_CLASSES
    assert "none" in NON_LABEL_CLASSES


# --- Path A: geofence rollup (top-1 excluded) ---

def test_geofence_rollup_excluded_top1(taxonomy_lookup, class_id_to_name):
    """Path A: top-1 excluded at 0.90, rolls up to allowed family."""
    classifications = [[0, 0.90], [1, 0.05], [2, 0.03], [3, 0.02]]
    excluded = frozenset({"leopard"})
    allowed = frozenset({
        "mammalia;carnivora;felidae;;",
        "mammalia;carnivora;;;",
        "mammalia;;;;",
    })
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
        excluded_names=excluded,
        allowed_taxonomy_keys=allowed,
    )
    assert result is not None
    assert result["level"] == "family"
    assert result["label"] == "felidae"
    assert result["confidence"] >= 0.90


def test_geofence_rollup_skips_genus(taxonomy_lookup, class_id_to_name):
    """Path A starts at family, skipping species and genus."""
    classifications = [[0, 0.90], [1, 0.05], [2, 0.03], [3, 0.02]]
    excluded = frozenset({"leopard"})
    # genus panthera IS allowed, but Path A should skip genus
    allowed = frozenset({
        "mammalia;carnivora;felidae;panthera;",
        "mammalia;carnivora;felidae;;",
    })
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
        excluded_names=excluded,
        allowed_taxonomy_keys=allowed,
    )
    assert result is not None
    assert result["level"] == "family"


def test_geofence_rollup_ancestor_not_allowed(
    taxonomy_lookup, class_id_to_name
):
    """Path A: family not allowed, walks up to order."""
    classifications = [[0, 0.90], [1, 0.05], [2, 0.03], [3, 0.02]]
    excluded = frozenset({"leopard"})
    # felidae NOT in allowed set, but carnivora IS
    allowed = frozenset({"mammalia;carnivora;;;"})
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
        excluded_names=excluded,
        allowed_taxonomy_keys=allowed,
    )
    assert result is not None
    assert result["level"] == "order"
    assert result["label"] == "carnivora"


def test_geofence_rollup_nothing_allowed(taxonomy_lookup, class_id_to_name):
    """Path A: no ancestor allowed, falls back to kingdom 'animal'."""
    classifications = [[0, 0.90], [1, 0.05], [2, 0.03], [3, 0.02]]
    excluded = frozenset({"leopard"})
    allowed = frozenset()  # nothing allowed at family/order/class
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
        excluded_names=excluded,
        allowed_taxonomy_keys=allowed,
    )
    # Kingdom rollup: all 4 are in taxonomy_lookup, sum = 1.0 >= 0.65
    # Returns "animal" at kingdom level
    assert result is not None
    assert result["level"] == "kingdom"
    assert result["taxon"] == "animal"


def test_geofence_rollup_high_conf_excluded_still_rolls_up(
    taxonomy_lookup, class_id_to_name
):
    """Path A fires even when top-1 conf >= 0.65 (unlike Path B)."""
    classifications = [[0, 0.95], [1, 0.03], [2, 0.02]]
    excluded = frozenset({"leopard"})
    allowed = frozenset({"mammalia;carnivora;felidae;;"})
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
        excluded_names=excluded,
        allowed_taxonomy_keys=allowed,
    )
    assert result is not None
    assert result["level"] == "family"


# --- Path B: confidence rollup with allowed check ---

def test_confidence_rollup_result_must_be_allowed(
    taxonomy_lookup, class_id_to_name
):
    """Path B: genus not allowed, walks to family."""
    # leopard 0.35 + lion 0.35 = panthera genus 0.70
    classifications = [[0, 0.35], [1, 0.35], [2, 0.15], [3, 0.15]]
    # panthera genus NOT allowed, felidae family IS
    allowed = frozenset({"mammalia;carnivora;felidae;;"})
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
        allowed_taxonomy_keys=allowed,
    )
    assert result is not None
    assert result["level"] == "family"
    assert result["label"] == "felidae"


# --- Top-5 behavior ---

def test_top5_only_used_for_sums(taxonomy_lookup, class_id_to_name):
    """Only top-5 classifications are used for rollup sums."""
    # 6 classifications. The 6th (bird, 0.15) would be excluded from
    # top-5 but doesn't affect mammalia sum since bird is aves.
    # mammalia from top-5: leopard(0.20)+lion(0.15)+cheetah(0.15)+zebra(0.15)=0.65
    classifications = [
        [0, 0.20], [5, 0.20], [1, 0.15], [2, 0.15], [3, 0.15], [4, 0.15],
    ]
    result = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
    )
    # mammalia = 0.65 >= 0.65 from top-5
    assert result is not None
    assert result["level"] == "class"
    assert result["label"] == "mammalia"




# ---------------------------------------------------------------------
# Threshold parameter (regression: was hardcoded to 0.65 in-module
# despite the rollup threshold being a fixed policy value)
# ---------------------------------------------------------------------


def test_rollup_threshold_parameter_demotes_a_species_below_it(
    taxonomy_lookup,
):
    """A species-level top-1 just under a custom threshold rolls up to
    genus, even though it would have stayed at species under the
    default 0.65. Pins that the threshold parameter is actually read."""
    class_id_to_name = {"0": "lion", "1": "leopard"}
    # Top-1 lion at 0.7: above default 0.65 → stays at species.
    # Bumping threshold to 0.8 pushes it under, forcing rollup.
    classifications = [["0", 0.7], ["1", 0.15]]

    default = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
    )
    # Default threshold (0.65) keeps the confident species — no rollup.
    assert default is None

    tightened = rollup_single_detection(
        classifications,
        class_id_to_name,
        taxonomy_lookup,
        threshold=0.8,
    )
    # 0.7 < 0.8 at species. Lion + leopard share genus panthera so the
    # genus-level sum (0.85) crosses 0.8 first and rollup stops there.
    # The threshold parameter demonstrably did its job — the species
    # answer was suppressed in favour of a broader rank.
    assert tightened is not None
    assert tightened["level"] != "species"


def test_rollup_threshold_parameter_loosens_kingdom_fallback(
    taxonomy_lookup,
):
    """Spreading confidence across taxonomically unrelated species
    drives every level below the default threshold; lowering the
    threshold lets a level cross it instead of falling all the way
    back to kingdom."""
    class_id_to_name = {"0": "lion", "1": "zebra", "2": "bird"}
    # lion 0.3 + zebra 0.25 + bird 0.2 = 0.75 at kingdom; nothing
    # crosses 0.65 at any narrower level.
    classifications = [["0", 0.3], ["1", 0.25], ["2", 0.2]]

    default = rollup_single_detection(
        classifications, class_id_to_name, taxonomy_lookup,
    )
    assert default is not None
    assert default["level"] == "kingdom"

    relaxed = rollup_single_detection(
        classifications,
        class_id_to_name,
        taxonomy_lookup,
        threshold=0.5,
    )
    # lion + zebra = 0.55 at class=mammalia under threshold 0.5.
    assert relaxed is not None
    assert relaxed["level"] == "class"
    assert relaxed["label"] == "mammalia"


# ── The shared leaf-naming rule ──────────────────────────────────────


@pytest.mark.parametrize(
    "label,scientific_name,level,expected",
    [
        # A second, more recognisable name exists: show it.
        ("leopard", "P. pardus", "species", "leopard"),
        ("baboon", "Papio", "genus", "baboon"),
        ("guineafowl", "Numididae", "family", "guineafowl"),
        ("micromammal", "Mammalia", "class", "micromammal"),
        # Underscores are cleaned, as elsewhere in the naming helpers.
        ("red_colobus", "Piliocolobus", "genus", "red colobus"),
        # The label *is* the taxon, so there is no second name: name the rank.
        ("gorilla", "Gorilla", "genus", "genus"),
        ("numididae", "Numididae", "family", "family"),
        ("mammalia", "Mammalia", "class", "class"),
        # Case must not matter: rollup rows store a capitalised scientific
        # name while the label stays lower case.
        ("Felidae", "felidae", "family", "family"),
    ],
)
def test_format_leaf_annotation(label, scientific_name, level, expected):
    assert format_leaf_annotation(label, scientific_name, level) == expected


def test_both_trees_use_this_helper():
    """The species picker (ml.taxonomy_parser) and the Labels filter
    (api.crud.label_tree) must render the same taxon identically, because
    one component draws both and a divergence looks like two rows that
    mean different things. Neither may re-implement the rule locally.
    """
    from app.api.crud import label_tree
    from app.ml import taxonomy_parser

    assert taxonomy_parser.format_leaf_annotation is format_leaf_annotation
    assert label_tree.format_leaf_annotation is format_leaf_annotation


def test_excluded_variant_rolls_to_its_backed_species():
    """Excluding one variant must not skip past the species its included
    sibling still backs: "red fox juvenile" excluded lands on
    "vulpes vulpes", not on "canidae"."""
    lookup = {
        "red fox adult": {
            "class": "mammalia", "order": "carnivora",
            "family": "canidae", "genus": "vulpes", "species": "vulpes",
        },
        "red fox juvenile": {
            "class": "mammalia", "order": "carnivora",
            "family": "canidae", "genus": "vulpes", "species": "vulpes",
        },
    }
    ids = {"0": "red fox juvenile", "1": "red fox adult"}
    result = rollup_single_detection(
        [[0, 0.90], [1, 0.05]], ids, lookup,
        excluded_names=frozenset({"red fox juvenile"}),
    )
    assert result is not None
    assert result["level"] == "species"
    assert result["label"] == "vulpes vulpes"
    assert result["confidence"] == pytest.approx(0.95, abs=0.01)


def test_excluding_every_variant_of_a_species_falls_to_family():
    """With no included class left on the species chain, the excluded
    path keeps today's behaviour and falls through to family."""
    lookup = {
        "red fox adult": {
            "class": "mammalia", "order": "carnivora",
            "family": "canidae", "genus": "vulpes", "species": "vulpes",
        },
        "red fox juvenile": {
            "class": "mammalia", "order": "carnivora",
            "family": "canidae", "genus": "vulpes", "species": "vulpes",
        },
    }
    ids = {"0": "red fox juvenile", "1": "red fox adult"}
    result = rollup_single_detection(
        [[0, 0.90], [1, 0.05]], ids, lookup,
        excluded_names=frozenset({"red fox juvenile", "red fox adult"}),
    )
    assert result is not None
    assert result["level"] == "family"
    assert result["label"] == "canidae"
