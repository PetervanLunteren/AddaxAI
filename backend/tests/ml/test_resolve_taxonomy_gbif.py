"""
Tests for scripts/resolve_taxonomy_gbif.py.

The script only runs at staging time, but the rules it encodes are subtle
and each one is here because real data broke it. Nothing hits the
network: GBIF records are stubbed, since the point is to pin our mapping
rules, not GBIF's answers.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from resolve_taxonomy_gbif import (  # noqa: E402
    Entry,
    _finest_legacy_name,
    record_to_ranks,
    resolve_entry,
)


def _record(**kw) -> dict:
    base = {
        "rank": "SPECIES",
        "class": "Mammalia",
        "order": "Carnivora",
        "family": "Felidae",
        "genus": "Panthera",
        "species": "Panthera pardus",
        "canonicalName": "Panthera pardus",
        "matchType": "EXACT",
        "confidence": 100,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------
# record_to_ranks: binomial splitting
# --------------------------------------------------------------------


def test_species_column_holds_the_epithet_only():
    ranks = record_to_ranks(_record())
    assert ranks["genus"] == "panthera"
    assert ranks["species"] == "pardus"


def test_subspecies_keeps_the_trinomial_tail():
    """Matches the shipped TKM-ADS-v1: `panthera,pardus saxicolor`."""
    ranks = record_to_ranks(
        _record(
            rank="SUBSPECIES",
            canonicalName="Panthera pardus saxicolor",
            species="Panthera pardus",
        )
    )
    assert ranks["genus"] == "panthera"
    assert ranks["species"] == "pardus saxicolor"


def test_repeated_genus_epithet_survives():
    """"Axis axis" must not collapse to an empty epithet."""
    ranks = record_to_ranks(
        _record(genus="Axis", species="Axis axis", canonicalName="Axis axis")
    )
    assert (ranks["genus"], ranks["species"]) == ("axis", "axis")


def test_genus_comes_from_the_binomial_not_the_genus_field():
    """
    The live SAH-DRY-ADS-v1 defect: GBIF returns canonicalName
    "Parahyaena brunnea" alongside genus "Hyaena". Trusting the genus
    field pairs it with a species of "parahyaena brunnea", which the UI
    renders as "H. parahyaena brunnea".
    """
    ranks = record_to_ranks(
        _record(
            genus="Hyaena",
            canonicalName="Parahyaena brunnea",
            species="Hyaena brunnea",
        )
    )
    assert ranks["genus"] == "parahyaena"
    assert ranks["species"] == "brunnea"


def test_ranks_above_species_leave_species_empty():
    ranks = record_to_ranks(
        _record(rank="GENUS", canonicalName="Panthera", species=None)
    )
    assert ranks["species"] == ""
    assert ranks["genus"] == "panthera"


def test_class_rank_record_fills_only_class():
    ranks = record_to_ranks(
        {
            "rank": "CLASS",
            "class": "Aves",
            "canonicalName": "Aves",
        }
    )
    assert ranks == {
        "class": "aves",
        "order": "",
        "family": "",
        "genus": "",
        "species": "",
    }


# --------------------------------------------------------------------
# The Reptilia rule
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "gbif_class", ["Squamata", "Testudines", "Crocodylia", "Rhynchocephalia"]
)
def test_gbif_reptile_classes_fold_under_reptilia(gbif_class: str):
    """
    GBIF's backbone has no Reptilia: it puts these at class rank with an
    empty order. AddaxAI shows a Linnaean tree, so they move to order.
    Reproduces the shipped SAH-DRY-ADS-v1 rows.
    """
    ranks = record_to_ranks(
        _record(rank="FAMILY", **{"class": gbif_class}, order=None, canonicalName="X")
    )
    assert ranks["class"] == "reptilia"
    assert ranks["order"] == gbif_class.lower()


def test_reptile_rule_matches_shipped_leopard_tortoise():
    ranks = record_to_ranks(
        {
            "rank": "SPECIES",
            "class": "Testudines",
            "order": None,
            "family": "Testudinidae",
            "genus": "Stigmochelys",
            "canonicalName": "Stigmochelys pardalis",
        }
    )
    assert ranks == {
        "class": "reptilia",
        "order": "testudines",
        "family": "testudinidae",
        "genus": "stigmochelys",
        "species": "pardalis",
    }


def test_amphibians_are_untouched():
    """GBIF has Amphibia as a class already, so no rule should fire."""
    ranks = record_to_ranks(
        _record(
            **{"class": "Amphibia"},
            order="Anura",
            family="Ranidae",
            genus="Lithobates",
            canonicalName="Lithobates catesbeianus",
        )
    )
    assert ranks["class"] == "amphibia"
    assert ranks["order"] == "anura"


def test_reptile_rule_does_not_clobber_an_existing_order():
    """Defensive: if GBIF ever starts filling order, keep it."""
    ranks = record_to_ranks(
        _record(**{"class": "Squamata"}, order="Serpentes", canonicalName="X y")
    )
    assert ranks["class"] == "squamata"
    assert ranks["order"] == "serpentes"


# --------------------------------------------------------------------
# _finest_legacy_name: reads all three legacy header shapes
# --------------------------------------------------------------------


def test_finest_legacy_name_strips_rank_prefix():
    row = {
        "level_class": "class Mammalia",
        "level_order": "order Carnivora",
        "level_species": "species Felis catus",
    }
    assert _finest_legacy_name(row) == "Felis catus"


def test_finest_legacy_name_reads_unprefixed_variant():
    """The DeepForestVision file carries no rank prefixes."""
    row = {
        "level_class": "mammalia",
        "level_order": "tubulidentata",
        "level_species": "orycteropus afer",
    }
    assert _finest_legacy_name(row) == "orycteropus afer"


def test_finest_legacy_name_falls_back_to_coarsest_present():
    row = {"level_class": "aves", "level_order": "", "level_species": ""}
    assert _finest_legacy_name(row) == "aves"


def test_finest_legacy_name_prefers_the_finest_rank():
    row = {"level_class": "mammalia", "level_genus": "papio", "level_species": ""}
    assert _finest_legacy_name(row) == "papio"


def test_finest_legacy_name_none_when_empty():
    assert _finest_legacy_name({"level_class": "", "level_species": ""}) is None
    assert _finest_legacy_name({}) is None


# --------------------------------------------------------------------
# resolve_entry: precedence and the model_class guard
# --------------------------------------------------------------------


def test_non_label_classes_get_an_empty_row_without_asking_gbif(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("GBIF must not be called for non-label classes")

    monkeypatch.setattr("resolve_taxonomy_gbif.resolve_by_key", boom)
    monkeypatch.setattr("resolve_taxonomy_gbif.resolve_by_name", boom)

    row, warning = resolve_entry(Entry("blank", "123", "Whatever"))
    assert row == {
        "model_class": "blank",
        "class": "",
        "order": "",
        "family": "",
        "genus": "",
        "species": "",
    }
    assert warning is None


def test_gbif_key_wins_over_scientific_name(monkeypatch):
    monkeypatch.setattr(
        "resolve_taxonomy_gbif.resolve_by_key", lambda k: _record()
    )
    monkeypatch.setattr(
        "resolve_taxonomy_gbif.resolve_by_name",
        lambda n: pytest.fail("name path must not run when a key is present"),
    )
    row, warning = resolve_entry(Entry("leopard", "5219426", "Something Else"))
    assert row["genus"] == "panthera"
    assert warning is None


def test_model_class_is_never_matched_against_gbif(monkeypatch):
    """
    The serval guard. GBIF matches the common name "serval" to a beetle
    genus in Mordellidae with an EXACT, full-confidence hit, so a row
    with no key and no scientific name must stay empty rather than guess.
    """
    monkeypatch.setattr(
        "resolve_taxonomy_gbif.resolve_by_name",
        lambda n: pytest.fail(f"must not resolve by model_class, got {n!r}"),
    )
    row, warning = resolve_entry(Entry("serval", None, None))
    assert row["class"] == ""
    assert warning == "no GBIF key and no scientific name to resolve"


def test_unresolvable_key_is_reported(monkeypatch):
    monkeypatch.setattr("resolve_taxonomy_gbif.resolve_by_key", lambda k: None)
    row, warning = resolve_entry(Entry("ghost", "999999999", None))
    assert row["class"] == ""
    assert "did not resolve" in warning


def test_synonym_reports_both_names(monkeypatch):
    monkeypatch.setattr(
        "resolve_taxonomy_gbif.resolve_by_key",
        lambda k: _record(
            status="SYNONYM",
            canonicalName="Felis lybica",
            species="Felis silvestris",
            genus="Felis",
        ),
    )
    row, warning = resolve_entry(Entry("african wild cat", "123", None))
    # The name the key points at is kept, not silently swapped.
    assert row["species"] == "lybica"
    assert "Felis lybica" in warning and "Felis silvestris" in warning


def test_fuzzy_name_match_is_reported(monkeypatch):
    monkeypatch.setattr(
        "resolve_taxonomy_gbif.resolve_by_name",
        lambda n: _record(matchType="FUZZY"),
    )
    _, warning = resolve_entry(Entry("leopard", None, "Panthera pardis"))
    assert "FUZZY" in warning


def test_match_without_a_class_is_reported(monkeypatch):
    """"arthropods" resolves to a phylum, which has no class."""
    monkeypatch.setattr(
        "resolve_taxonomy_gbif.resolve_by_name",
        lambda n: {
            "rank": "PHYLUM",
            "canonicalName": "Arthropoda",
            "matchType": "EXACT",
        },
    )
    row, warning = resolve_entry(Entry("arthropods", None, "Arthropoda"))
    assert row["class"] == ""
    assert "no class" in warning
