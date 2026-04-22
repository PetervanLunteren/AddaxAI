"""
Pure-math tests for confusion matrix and classification report.

Exercises the helpers in app.api.crud.performance that don't touch the
DB: top-N folding, macro / weighted averages, F1 harmonic mean, class
ordering, rank value resolution.
"""

from collections import Counter

import pytest

from app.api.crud.performance import (
    DETECTOR_CATEGORIES,
    OTHER_BUCKET,
    _apply_top_n,
    _harmonic_mean,
    _macro,
    _ordered_classes,
    _taxon_at_rank,
    _weighted,
)


class _Row:
    """Minimal stand-in for a LabelTaxonomy row."""

    def __init__(
        self,
        name: str,
        *,
        taxon_class: str | None = None,
        taxon_order: str | None = None,
        taxon_family: str | None = None,
        taxon_genus: str | None = None,
        taxon_species: str | None = None,
    ) -> None:
        self.name = name
        self.taxon_class = taxon_class
        self.taxon_order = taxon_order
        self.taxon_family = taxon_family
        self.taxon_genus = taxon_genus
        self.taxon_species = taxon_species


# ---------------------------------------------------------------------------
# _taxon_at_rank
# ---------------------------------------------------------------------------


def test_taxon_at_rank_reads_family_column() -> None:
    row = _Row("leopard", taxon_family="felidae", taxon_genus="panthera")
    assert _taxon_at_rank(row, "family") == "felidae"
    assert _taxon_at_rank(row, "genus") == "panthera"


def test_taxon_at_rank_species_uses_unique_leaf_name() -> None:
    row = _Row("panthera_pardus", taxon_species="pardus", taxon_genus="panthera")
    # species rank returns row.name, not taxon_species, so it doesn't
    # collide across genera with the same species epithet
    assert _taxon_at_rank(row, "species") == "panthera_pardus"


def test_taxon_at_rank_missing_column_returns_none() -> None:
    row = _Row("bird", taxon_class="aves")
    assert _taxon_at_rank(row, "family") is None


def test_taxon_at_rank_none_row() -> None:
    assert _taxon_at_rank(None, "family") is None


# ---------------------------------------------------------------------------
# _ordered_classes
# ---------------------------------------------------------------------------


def test_ordered_classes_puts_detector_head_first() -> None:
    all_classes = {"wolf", "deer", "person", "animal", "vehicle"}
    totals = {"wolf": 10, "deer": 30, "person": 5, "animal": 2, "vehicle": 1}
    ordered = _ordered_classes(all_classes, totals)
    # animal, person, vehicle always first in fixed order
    assert ordered[:3] == ["animal", "person", "vehicle"]
    # remaining classes sorted by descending support
    assert ordered[3:] == ["deer", "wolf"]


def test_ordered_classes_omits_missing_detector_categories() -> None:
    all_classes = {"wolf", "deer"}
    totals = {"wolf": 3, "deer": 2}
    ordered = _ordered_classes(all_classes, totals)
    assert ordered == ["wolf", "deer"]
    for c in DETECTOR_CATEGORIES:
        assert c not in ordered


def test_ordered_classes_alphabetical_tiebreaker() -> None:
    all_classes = {"alpha", "beta", "gamma"}
    totals = {"alpha": 5, "beta": 5, "gamma": 5}
    ordered = _ordered_classes(all_classes, totals)
    assert ordered == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# _apply_top_n
# ---------------------------------------------------------------------------


def test_apply_top_n_noop_when_under_limit() -> None:
    ordered = ["a", "b", "c"]
    counts: Counter = Counter({("a", "a"): 3, ("b", "a"): 1})
    totals = {"a": 4, "b": 1, "c": 0}
    new_ordered, new_counts, other = _apply_top_n(ordered, counts, totals, 10)
    assert new_ordered == ordered
    assert new_counts == counts
    assert other is False


def test_apply_top_n_folds_tail_into_other() -> None:
    ordered = ["a", "b", "c", "d", "e"]
    counts: Counter = Counter(
        {
            ("a", "a"): 10,
            ("b", "b"): 8,
            ("c", "d"): 2,   # c confuses with d
            ("d", "c"): 3,
            ("e", "a"): 1,
        }
    )
    totals = {"a": 10, "b": 8, "c": 2, "d": 3, "e": 1}
    new_ordered, new_counts, other = _apply_top_n(ordered, counts, totals, 2)
    assert new_ordered == ["a", "b", OTHER_BUCKET]
    assert other is True
    # c, d, e collapse into "other" on both axes; totals must be preserved
    assert sum(new_counts.values()) == sum(counts.values())
    # a row stays as-is
    assert new_counts[("a", "a")] == 10
    # e row collapses to other row, and predicts a
    assert new_counts[(OTHER_BUCKET, "a")] == 1
    # c and d get folded entirely into (other, other) because they only
    # confused with each other
    assert new_counts[(OTHER_BUCKET, OTHER_BUCKET)] == 5


def test_apply_top_n_keeps_detector_head_within_budget() -> None:
    ordered = ["animal", "person", "wolf", "deer", "bear"]
    counts: Counter = Counter(
        {
            ("wolf", "wolf"): 20,
            ("deer", "deer"): 15,
            ("bear", "bear"): 10,
            ("animal", "animal"): 2,
            ("person", "person"): 1,
        }
    )
    totals = {"wolf": 20, "deer": 15, "bear": 10, "animal": 2, "person": 1}
    # top_n=3 but detector head has 2 entries → only 1 species slot left
    new_ordered, _new_counts, other = _apply_top_n(ordered, counts, totals, 3)
    assert new_ordered[:2] == ["animal", "person"]
    assert "wolf" in new_ordered  # best species by support
    assert OTHER_BUCKET in new_ordered
    assert other is True


def test_apply_top_n_none_disables_collapse() -> None:
    ordered = ["a", "b", "c", "d"]
    counts: Counter = Counter({("a", "a"): 1, ("b", "b"): 1, ("c", "c"): 1, ("d", "d"): 1})
    totals = {"a": 1, "b": 1, "c": 1, "d": 1}
    new_ordered, new_counts, other = _apply_top_n(ordered, counts, totals, None)
    assert new_ordered == ordered
    assert new_counts == counts
    assert other is False


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------


def test_harmonic_mean_typical() -> None:
    assert _harmonic_mean(0.6, 0.4) == pytest.approx(0.48)


def test_harmonic_mean_zero_sum_returns_none() -> None:
    assert _harmonic_mean(0.0, 0.0) is None


def test_harmonic_mean_none_input_returns_none() -> None:
    assert _harmonic_mean(None, 0.5) is None
    assert _harmonic_mean(0.5, None) is None


def test_macro_ignores_none() -> None:
    assert _macro([1.0, None, 0.5]) == pytest.approx(0.75)


def test_macro_all_none_returns_none() -> None:
    assert _macro([None, None]) is None


def test_weighted_support_weighted() -> None:
    # two classes with f1=1.0 and 0.0, supports 9 and 1 → weighted 0.9
    assert _weighted([1.0, 0.0], [9, 1]) == pytest.approx(0.9)


def test_weighted_none_values_skipped() -> None:
    # the class with None is ignored entirely, including its weight
    assert _weighted([None, 0.5], [10, 2]) == pytest.approx(0.5)


def test_weighted_all_none_returns_none() -> None:
    assert _weighted([None, None], [5, 5]) is None


def test_weighted_zero_support_returns_none() -> None:
    assert _weighted([0.5, 0.5], [0, 0]) is None
