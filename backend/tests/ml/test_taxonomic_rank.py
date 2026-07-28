"""
Tests for the shared taxonomic_rank resolver.

Exercises resolve_rank() directly; no DB. The dashboard's SQL CASE in
statistics._rank_display_label is expected to produce the same output
for the same inputs — these tests are the contract both impls satisfy.
"""

from dataclasses import dataclass

import pytest

from app.ml.taxonomic_rank import (
    HIGHER_LEVEL_TAXA,
    NO_TAXONOMY,
    RANK_OPTIONS,
    resolve_rank,
)


@dataclass
class Row:
    """Minimal stand-in for a LabelTaxonomy row."""

    name: str = ""
    scientific_name: str | None = None
    taxon_class: str | None = None
    taxon_order: str | None = None
    taxon_family: str | None = None
    taxon_genus: str | None = None
    taxon_species: str | None = None


# ---------------------------------------------------------------------------
# Non-animal categories always return the category
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", ["all", "class", "order", "family", "genus", "species"])
def test_person_returns_category_at_every_rank(rank):
    assert resolve_rank(
        category="person", label=None, scientific_name=None, taxonomy_row=None, rank=rank,
    ) == "person"


@pytest.mark.parametrize("rank", ["all", "class", "order", "family", "genus", "species"])
def test_vehicle_returns_category_at_every_rank(rank):
    assert resolve_rank(
        category="vehicle", label=None, scientific_name=None, taxonomy_row=None, rank=rank,
    ) == "vehicle"


# ---------------------------------------------------------------------------
# Most-specific mode
# ---------------------------------------------------------------------------


def test_all_mode_prefers_scientific_name():
    assert resolve_rank(
        category="animal",
        label="leopard",
        scientific_name="P. pardus",
        taxonomy_row=Row(name="leopard", scientific_name="P. pardus", taxon_species="pardus"),
        rank="all",
    ) == "P. pardus"


def test_all_mode_falls_back_to_label_then_category():
    # No scientific_name, has label
    assert resolve_rank(
        category="animal", label="deer", scientific_name=None, taxonomy_row=None, rank="all",
    ) == "deer"
    # No label at all: animal detector-only detection
    assert resolve_rank(
        category="animal", label=None, scientific_name=None, taxonomy_row=None, rank="all",
    ) == "animal"


def test_none_rank_behaves_as_all():
    assert resolve_rank(
        category="animal", label="deer", scientific_name=None, taxonomy_row=None, rank=None,
    ) == "deer"


# ---------------------------------------------------------------------------
# Specific rank with complete taxonomy
# ---------------------------------------------------------------------------


def test_species_rank_uses_scientific_name():
    row = Row(
        name="leopard",
        scientific_name="P. pardus",
        taxon_class="mammalia",
        taxon_order="carnivora",
        taxon_family="felidae",
        taxon_genus="panthera",
        taxon_species="pardus",
    )
    assert resolve_rank(
        category="animal", label="leopard", scientific_name="P. pardus",
        taxonomy_row=row, rank="species",
    ) == "P. pardus"


def test_species_rank_falls_back_to_name_when_no_scientific_name():
    row = Row(name="leopard", taxon_species="pardus")
    assert resolve_rank(
        category="animal", label="leopard", scientific_name=None,
        taxonomy_row=row, rank="species",
    ) == "leopard"


def test_family_rank_returns_capitalised_taxon_family():
    row = Row(
        name="leopard",
        taxon_class="mammalia",
        taxon_family="felidae",
        taxon_genus="panthera",
        taxon_species="pardus",
    )
    # Family / genus / order / class are stored lowercase by the CSV
    # importer; resolve_rank capitalises for display.
    assert resolve_rank(
        category="animal", label="leopard", scientific_name=None,
        taxonomy_row=row, rank="family",
    ) == "Felidae"


def test_class_rank_returns_capitalised_taxon_class():
    row = Row(name="leopard", taxon_class="mammalia", taxon_family="felidae")
    assert resolve_rank(
        category="animal", label="leopard", scientific_name=None,
        taxonomy_row=row, rank="class",
    ) == "Mammalia"


# ---------------------------------------------------------------------------
# Bucket behaviour
# ---------------------------------------------------------------------------


def test_rollup_row_at_species_becomes_higher_level_taxa():
    # Family-level rollup entry (taxon_species is NULL) asked at species rank.
    # This is the exact case that caused confusing rows in the matrix.
    row = Row(
        name="Equidae",
        taxon_class="mammalia",
        taxon_order="perissodactyla",
        taxon_family="Equidae",
    )
    assert resolve_rank(
        category="animal", label="Equidae", scientific_name=None,
        taxonomy_row=row, rank="species",
    ) == HIGHER_LEVEL_TAXA


def test_class_only_row_at_family_becomes_higher_level_taxa():
    row = Row(name="bird", taxon_class="aves")
    assert resolve_rank(
        category="animal", label="bird", scientific_name=None,
        taxonomy_row=row, rank="family",
    ) == HIGHER_LEVEL_TAXA


def test_no_taxonomy_row_at_specific_rank_becomes_no_taxonomy():
    assert resolve_rank(
        category="animal", label="bait", scientific_name=None,
        taxonomy_row=None, rank="species",
    ) == NO_TAXONOMY


def test_all_null_taxonomy_row_becomes_no_taxonomy():
    # A row with no taxonomy fields at all (e.g. a custom user label
    # with no taxonomy info). taxon_class is None → bucket.
    row = Row(name="custom_label")
    assert resolve_rank(
        category="animal", label="custom_label", scientific_name=None,
        taxonomy_row=row, rank="genus",
    ) == NO_TAXONOMY


# ---------------------------------------------------------------------------
# Option list contract
# ---------------------------------------------------------------------------


def test_rank_options_first_is_all():
    assert RANK_OPTIONS[0][0] == "all"
    assert RANK_OPTIONS[0][1] == "Most specific"


def test_rank_options_cover_every_real_rank():
    values = [v for v, _ in RANK_OPTIONS]
    for rank in ("class", "order", "family", "genus", "species"):
        assert rank in values
