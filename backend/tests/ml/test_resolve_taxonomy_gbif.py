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
    ("gbif_class", "real_class"),
    [
        ("Squamata", "reptilia"),
        ("Testudines", "reptilia"),
        ("Crocodylia", "reptilia"),
        ("Rhynchocephalia", "reptilia"),
        ("Anura", "amphibia"),
        ("Caudata", "amphibia"),
        ("Urodela", "amphibia"),
        ("Gymnophiona", "amphibia"),
    ],
)
def test_order_as_class_folds_under_the_real_class(gbif_class: str, real_class: str):
    """
    GBIF's backbone has no Reptilia and returns some amphibian orders at
    class rank too. AddaxAI shows a Linnaean tree, so they move to order.
    Mirrors ORDER_AS_CLASS in cls-training-pipeline/taxon-mapping.
    """
    ranks = record_to_ranks(
        _record(rank="FAMILY", **{"class": gbif_class}, order=None, canonicalName="Xidae")
    )
    assert ranks["class"] == real_class
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


def test_amphibians_already_correct_are_untouched():
    """Most amphibian keys come back right; no rule should fire on those."""
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


def test_record_rank_beats_gbif_rank_fields():
    """
    GBIF key 12170551 is the project's "unknown reptile": rank=CLASS,
    canonicalName=Reptilia, but class=Squamata. Trusting the field would
    claim every unidentified reptile is a squamate rather than a turtle,
    and would then wrongly trip ORDER_AS_CLASS on the way out.
    """
    ranks = record_to_ranks(
        {
            "rank": "CLASS",
            "canonicalName": "Reptilia",
            "class": "Squamata",
            "order": None,
        }
    )
    assert ranks["class"] == "reptilia"
    assert ranks["order"] == ""


@pytest.mark.parametrize(
    ("rank", "column"),
    [("CLASS", "class"), ("ORDER", "order"), ("FAMILY", "family"), ("GENUS", "genus")],
)
def test_canonical_name_owns_the_records_own_rank(rank: str, column: str):
    ranks = record_to_ranks(
        {"rank": rank, "canonicalName": "Correctus", column: "Wrongus"}
    )
    assert ranks[column] == "correctus"


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


# --------------------------------------------------------------------
# Literature overrides: GBIF helps, the literature decides
# --------------------------------------------------------------------


def test_full_override_skips_gbif_entirely(monkeypatch):
    """
    The Neogale case. The 2021 split moved the American mink to Neogale,
    which GBIF still resolves as a genus-rank synonym of Mustela with no
    species at all, so the row is written by hand.
    """
    def boom(*a, **k):
        raise AssertionError("a fully hand-written row must not call GBIF")

    monkeypatch.setattr("resolve_taxonomy_gbif.resolve_by_key", boom)
    monkeypatch.setattr("resolve_taxonomy_gbif.resolve_by_name", boom)

    row, warning = resolve_entry(
        Entry(
            "american mink",
            None,
            "Neogale vison",
            {
                "class": "mammalia",
                "order": "carnivora",
                "family": "mustelidae",
                "genus": "neogale",
                "species": "vison",
            },
        )
    )
    assert row == {
        "model_class": "american mink",
        "class": "mammalia",
        "order": "carnivora",
        "family": "mustelidae",
        "genus": "neogale",
        "species": "vison",
    }
    assert "literature override" in warning


def test_partial_override_lets_gbif_fill_the_rest(monkeypatch):
    monkeypatch.setattr(
        "resolve_taxonomy_gbif.resolve_by_name", lambda n: _record()
    )
    row, warning = resolve_entry(
        Entry("leopard", None, "Panthera pardus", {"genus": "neofelis"})
    )
    assert row["genus"] == "neofelis"  # hand-written wins
    assert row["family"] == "felidae"  # GBIF fills the gap
    assert row["species"] == "pardus"
    assert "literature override on genus" in warning


def test_override_is_always_reported_for_review():
    """An override is a deliberate divergence; never let it pass silently."""
    _, warning = resolve_entry(
        Entry("x", None, None, dict.fromkeys(("class", "order", "family", "genus", "species"), "z"))
    )
    assert warning is not None


def test_non_label_class_beats_an_override():
    """"blank" carries no taxonomy even if someone hand-writes one."""
    row, warning = resolve_entry(Entry("blank", None, None, {"class": "mammalia"}))
    assert row["class"] == ""
    assert warning is None


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
