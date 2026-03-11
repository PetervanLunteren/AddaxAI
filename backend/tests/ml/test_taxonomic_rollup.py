"""Tests for app.ml.taxonomic_rollup."""

import csv
import tempfile
from pathlib import Path

import pytest

from app.ml.label_exclusion import NON_LABEL_CLASSES
from app.ml.taxonomic_rollup import (
    apply_taxonomic_rollup_to_results,
    load_taxonomy_lookup,
    rollup_single_detection,
)

SAMPLE_TAXONOMY_ROWS = [
    {"model_class": "leopard", "class": "mammalia", "order": "carnivora", "family": "felidae", "genus": "panthera", "species": "pardus"},
    {"model_class": "lion", "class": "mammalia", "order": "carnivora", "family": "felidae", "genus": "panthera", "species": "leo"},
    {"model_class": "cheetah", "class": "mammalia", "order": "carnivora", "family": "felidae", "genus": "acinonyx", "species": "jubatus"},
    {"model_class": "zebra", "class": "mammalia", "order": "perissodactyla", "family": "equidae", "genus": "equus", "species": "quagga"},
    {"model_class": "bird", "class": "aves", "order": "", "family": "", "genus": "", "species": ""},
    {"model_class": "blank", "class": "", "order": "", "family": "", "genus": "", "species": ""},
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


def test_skip_non_taxonomic(taxonomy_lookup, class_id_to_name):
    # top-1 is "blank" (not in taxonomy) → skip
    classifications = [[5, 0.50], [0, 0.30], [1, 0.20]]
    result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
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
                    # Confident → skip
                    {
                        "bbox": [0.1, 0.1, 0.5, 0.5],
                        "classifications": [[0, 0.80], [1, 0.20]],
                    },
                    # Non-taxonomic → skip
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

    # Second: unchanged (non-taxonomic top-1)
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


def test_non_label_not_in_taxonomy_skipped(taxonomy_lookup, class_id_to_name):
    """Classes not in taxonomy_lookup are skipped by rollup (including non-label classes)."""
    # "blank" (class_id 5) is not in taxonomy_lookup -> rollup skips it
    classifications = [[5, 0.50], [0, 0.30], [1, 0.20]]
    result = rollup_single_detection(classifications, class_id_to_name, taxonomy_lookup)
    assert result is None


def test_non_label_classes_constant():
    """NON_LABEL_CLASSES (now in label_exclusion.py) contains the expected entries."""
    assert "blank" in NON_LABEL_CLASSES
    assert "empty" in NON_LABEL_CLASSES
    assert "false detection" in NON_LABEL_CLASSES
    assert "none" in NON_LABEL_CLASSES
