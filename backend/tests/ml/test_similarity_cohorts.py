"""
Tests for similarity_script cohort grouping and descendant filter.

The script runs as a subprocess in env-addaxai-base for production use,
but its grouping pass (`_group_cohorts`) and neighbour signal pass
(`_compute_neighbor_signals`) are pure Python helpers that import
cleanly into the main venv. FAISS is not needed here: `_group_cohorts`
takes pre-computed signals, and the descendant-filter test injects a
fake index with a hard-coded neighbour table.
"""

import numpy as np

from app.ml.inference.similarity_script import (
    _compute_neighbor_signals,
    _group_cohorts,
    _is_useful_suggestion,
)


def _meta(
    label: str | None,
    *,
    scientific_name: str | None = None,
    category: str | None = "animal",
    verified: bool = False,
    suggestion_dismissed: bool = False,
) -> dict:
    return {
        "label": label,
        "scientific_name": scientific_name,
        "category": category,
        "verified": verified,
        "suggestion_dismissed": suggestion_dismissed,
    }


# ── _is_useful_suggestion ────────────────────────────────────────────────


def test_useful_suggestion_keeps_class_to_species():
    # Current = class, suggested = species: more specific, allowed.
    current = {"level": "class", "taxon_class": "aves"}
    suggested = {
        "level": "species",
        "taxon_class": "aves",
        "taxon_order": "passeriformes",
        "taxon_family": "corvidae",
        "taxon_genus": "corvus",
        "taxon_species": "brachyrhynchos",
    }
    assert _is_useful_suggestion(suggested, current)


def test_useful_suggestion_keeps_genus_to_species():
    current = {
        "level": "genus",
        "taxon_class": "mammalia",
        "taxon_order": "carnivora",
        "taxon_family": "canidae",
        "taxon_genus": "canis",
    }
    suggested = {
        "level": "species",
        "taxon_class": "mammalia",
        "taxon_order": "carnivora",
        "taxon_family": "canidae",
        "taxon_genus": "canis",
        "taxon_species": "familiaris",
    }
    assert _is_useful_suggestion(suggested, current)


def test_useful_suggestion_drops_ancestor_direction():
    # Current = species, suggested = family (its parent rank). Broader,
    # blocked.
    current = {
        "level": "species",
        "taxon_class": "aves",
        "taxon_order": "galliformes",
        "taxon_family": "phasianidae",
        "taxon_genus": "gallus",
        "taxon_species": "gallus domesticus",
    }
    suggested = {
        "level": "family",
        "taxon_class": "aves",
        "taxon_order": "galliformes",
        "taxon_family": "phasianidae",
    }
    assert not _is_useful_suggestion(suggested, current)


def test_useful_suggestion_keeps_lateral_same_family():
    # grey fox (urocyon) vs coyote (canis): same family, different
    # genera, sibling species. Same rank, allowed (this is the case
    # the user most often needs to fix).
    current = {
        "level": "species",
        "taxon_class": "mammalia",
        "taxon_order": "carnivora",
        "taxon_family": "canidae",
        "taxon_genus": "urocyon",
        "taxon_species": "cinereoargenteus",
    }
    suggested = {
        "level": "species",
        "taxon_class": "mammalia",
        "taxon_order": "carnivora",
        "taxon_family": "canidae",
        "taxon_genus": "canis",
        "taxon_species": "latrans",
    }
    assert _is_useful_suggestion(suggested, current)


def test_useful_suggestion_keeps_cross_branch_same_rank():
    # Same rank (species) in totally different branches. Allowed in
    # principle; min_count handles the noise.
    current = {"level": "species", "taxon_class": "aves"}
    suggested = {"level": "species", "taxon_class": "mammalia"}
    assert _is_useful_suggestion(suggested, current)


def test_useful_suggestion_drops_missing_taxonomy():
    assert not _is_useful_suggestion(None, {"level": "class", "taxon_class": "aves"})
    assert not _is_useful_suggestion({"level": "species"}, None)
    assert not _is_useful_suggestion(None, None)


# ── _group_cohorts ───────────────────────────────────────────────────────


def test_group_cohorts_groups_by_triple_key():
    det_ids = ["a", "b", "c", "d", "e"]
    metas = [
        _meta("aves"),
        _meta("aves"),
        _meta("aves"),
        _meta("canis"),
        _meta("canis"),
    ]
    top_labels = [
        "american crow",
        "american crow",
        "american crow",
        "domestic dog",
        "domestic dog",
    ]
    agreement = np.array([0.1, 0.2, 0.3, 0.1, 0.2], dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)

    by_pair = {(r["current_label"], r["suggested_label"]): r for r in result}
    assert len(by_pair) == 2
    assert by_pair[("aves", "american crow")]["count"] == 3
    assert by_pair[("canis", "domestic dog")]["count"] == 2


def test_group_cohorts_separates_by_category():
    # Same labels but different category → different cohorts.
    det_ids = ["a", "b", "c"]
    metas = [
        _meta("aves", category="animal"),
        _meta("aves", category="animal"),
        _meta("aves", category="person"),
    ]
    top_labels = ["american crow"] * 3
    agreement = np.zeros(3, dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    counts = sorted(r["count"] for r in result)
    assert counts == [1, 2]


def test_group_cohorts_orders_by_count_desc():
    det_ids = [f"id{i}" for i in range(10)]
    metas = [_meta("aves")] * 8 + [_meta("canis")] * 2
    top_labels = ["american crow"] * 8 + ["domestic dog"] * 2
    agreement = np.zeros(10, dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    assert [r["suggested_label"] for r in result] == ["american crow", "domestic dog"]


def test_group_cohorts_min_count_drops_small():
    det_ids = [f"id{i}" for i in range(10)]
    metas = [_meta("aves")] * 5 + [_meta("canis")] * 5
    top_labels = ["american crow"] * 5 + ["domestic dog"] * 5
    agreement = np.zeros(10, dtype=np.float32)

    assert _group_cohorts(det_ids, metas, agreement, top_labels, 6, 10) == []
    assert len(_group_cohorts(det_ids, metas, agreement, top_labels, 5, 10)) == 2


def test_group_cohorts_excludes_verified():
    det_ids = ["a", "b"]
    metas = [_meta("aves", verified=True), _meta("aves", verified=False)]
    top_labels = ["american crow", "american crow"]
    agreement = np.zeros(2, dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    assert len(result) == 1
    assert result[0]["count"] == 1
    assert result[0]["detection_ids"] == ["b"]


def test_group_cohorts_excludes_dismissed():
    # A dismissed crop is skipped as a cohort member, exactly like a
    # verified one. The remaining members still form the cohort.
    det_ids = ["a", "b"]
    metas = [
        _meta("aves", suggestion_dismissed=True),
        _meta("aves", suggestion_dismissed=False),
    ]
    top_labels = ["american crow", "american crow"]
    agreement = np.zeros(2, dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    assert len(result) == 1
    assert result[0]["count"] == 1
    assert result[0]["detection_ids"] == ["b"]


def test_group_cohorts_excludes_detections_with_no_suggestion():
    det_ids = ["a", "b"]
    metas = [_meta("aves"), _meta("aves")]
    top_labels = [None, "american crow"]
    agreement = np.zeros(2, dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    assert len(result) == 1
    assert result[0]["detection_ids"] == ["b"]


def test_group_cohorts_sorts_members_by_agreement_asc():
    det_ids = ["high", "low", "mid"]
    metas = [_meta("aves")] * 3
    top_labels = ["american crow"] * 3
    agreement = np.array([0.6, 0.1, 0.3], dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    assert result[0]["detection_ids"] == ["low", "mid", "high"]


def test_group_cohorts_max_cohorts_cap():
    # Five cohorts of size 3 each.
    det_ids = [f"id{i}" for i in range(15)]
    metas = []
    top_labels = []
    for cohort in range(5):
        for _ in range(3):
            metas.append(_meta(f"current{cohort}"))
            top_labels.append(f"suggested{cohort}")
    agreement = np.zeros(15, dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 2)
    assert len(result) == 2


def test_group_cohorts_carries_scientific_names():
    det_ids = ["a", "b"]
    metas = [
        _meta("aves", scientific_name="Aves"),
        # b is in the dataset so 'american crow' has a known display
        # name; b itself does not contribute a cohort (top_labels[1] is
        # None) so it cannot pollute the output.
        _meta("american crow", scientific_name="C. brachyrhynchos"),
    ]
    top_labels = ["american crow", None]
    agreement = np.array([0.1, 1.0], dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    assert result[0]["current_scientific_name"] == "Aves"
    assert result[0]["suggested_scientific_name"] == "C. brachyrhynchos"


def test_group_cohorts_handles_none_current_label():
    det_ids = ["a", "b"]
    metas = [_meta(None), _meta(None)]
    top_labels = ["domestic dog", "domestic dog"]
    agreement = np.zeros(2, dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    assert len(result) == 1
    assert result[0]["current_label"] is None
    assert result[0]["count"] == 2


def test_group_cohorts_carries_current_label_taxonomy_id():
    """The taxonomy id of the current label travels with the cohort.

    The frontend's "Review crops" navigation uses that id to drop the
    user into the existing Observations label filter (keyed on taxonomy
    id, not label name).
    """
    det_ids = ["a", "b"]
    metas = [
        {**_meta("aves"), "label_taxonomy_id": "tax-aves-uuid"},
        {**_meta("aves"), "label_taxonomy_id": "tax-aves-uuid"},
    ]
    top_labels = ["american crow", "american crow"]
    agreement = np.zeros(2, dtype=np.float32)
    result = _group_cohorts(det_ids, metas, agreement, top_labels, 1, 10)
    assert result[0]["current_label_taxonomy_id"] == "tax-aves-uuid"


# ── _compute_neighbor_signals descendant filter ──────────────────────────


class _FakeIndex:
    """Stand-in for faiss.IndexFlatIP with a hard-coded neighbour table.

    `search` ignores the query vectors and returns the rows of `nbrs`
    truncated to the requested `k`. Similarities are not used by the
    helper so we return zeros.
    """

    def __init__(self, nbrs: np.ndarray):
        self._nbrs = np.asarray(nbrs)

    def search(self, vectors, k):
        n, available = self._nbrs.shape
        k_eff = min(k, available)
        return (
            np.zeros((n, k_eff), dtype=np.float32),
            self._nbrs[:, :k_eff],
        )


def test_compute_neighbor_signals_keeps_descendant_and_lateral_drops_broader(
    monkeypatch,
):
    """Descendant and same-rank suggestions surface; broader-rank ones do not."""
    # 33 detections total:
    #   0: canis with 10 'domestic dog' neighbours → descendant, kept.
    #   1: 'grey fox' with 10 'coyote' neighbours → same-rank lateral, kept.
    #   2: 'domestic dog' with 10 'canidae' neighbours → broader, dropped.
    #   3-12: 'domestic dog' fillers.
    #  13-22: 'coyote' fillers.
    #  23-32: 'canidae' fillers.
    full_label_list = (
        ["canis", "grey fox", "domestic dog"]
        + ["domestic dog"] * 10
        + ["coyote"] * 10
        + ["canidae"] * 10
    )
    nbrs = np.zeros((33, 11), dtype=np.int64)
    nbrs[0] = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    nbrs[1] = [1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    nbrs[2] = [2, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    for i in range(3, 33):
        nbrs[i] = [i] * 11

    monkeypatch.setattr(
        "app.ml.inference.similarity_script._load_label_taxonomy",
        lambda db_path, project_id, labels: {
            "canidae": {
                "level": "family",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
            },
            "canis": {
                "level": "genus",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
                "taxon_genus": "canis",
            },
            "domestic dog": {
                "level": "species",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
                "taxon_genus": "canis",
                "taxon_species": "familiaris",
            },
            "grey fox": {
                "level": "species",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
                "taxon_genus": "urocyon",
                "taxon_species": "cinereoargenteus",
            },
            "coyote": {
                "level": "species",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
                "taxon_genus": "canis",
                "taxon_species": "latrans",
            },
        },
    )

    vectors = np.zeros((33, 4), dtype=np.float32)
    index = _FakeIndex(nbrs)
    _, top = _compute_neighbor_signals(
        index, vectors, full_label_list, "/dummy.db", "proj"
    )

    # canis (genus) → domestic dog (species): more specific, kept.
    assert top[0] == "domestic dog"
    # grey fox (species) → coyote (species): same rank, kept.
    assert top[1] == "coyote"
    # domestic dog (species) → canidae (family): broader, dropped.
    assert top[2] is None


def test_compute_neighbor_signals_agreement_score_on_perfect_match(monkeypatch):
    # All five detections are 'aves' and all neighbours are 'aves' too →
    # agreement = 1.0 for everyone, no suggestion offered.
    labels = ["aves"] * 5
    nbrs = np.array(
        [
            [i, *[(i + 1 + j) % 5 for j in range(10)]]
            for i in range(5)
        ],
        dtype=np.int64,
    )

    monkeypatch.setattr(
        "app.ml.inference.similarity_script._load_label_taxonomy",
        lambda db_path, project_id, ls: {
            "aves": {"level": "class", "taxon_class": "aves"},
        },
    )

    vectors = np.zeros((5, 4), dtype=np.float32)
    index = _FakeIndex(nbrs)
    agreement, top = _compute_neighbor_signals(
        index, vectors, labels, "/dummy.db", "proj"
    )

    assert all(a == 1.0 for a in agreement)
    assert all(t is None for t in top)


def test_compute_neighbor_signals_drops_weak_plurality(monkeypatch):
    """A plurality below NEIGHBOR_MAJORITY_FRACTION does not surface.

    Detection 0 = "canis" with neighbours: 5 'domestic dog',
    3 'coyote', 2 'urocyon'. Top label is 'domestic dog' with 5/10 =
    0.5, which is below the 0.6 threshold. No suggestion should
    surface even though 'domestic dog' is a clean descendant.
    """
    labels = ["canis"] + ["domestic dog"] * 5 + ["coyote"] * 3 + ["urocyon"] * 2
    nbrs = np.zeros((11, 11), dtype=np.int64)
    nbrs[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(1, 11):
        nbrs[i] = [i] * 11

    monkeypatch.setattr(
        "app.ml.inference.similarity_script._load_label_taxonomy",
        lambda db_path, project_id, ls: {
            "canis": {
                "level": "genus",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
                "taxon_genus": "canis",
            },
            "domestic dog": {
                "level": "species",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
                "taxon_genus": "canis",
                "taxon_species": "familiaris",
            },
            "coyote": {
                "level": "species",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
                "taxon_genus": "canis",
                "taxon_species": "latrans",
            },
            "urocyon": {
                "level": "genus",
                "taxon_class": "mammalia",
                "taxon_order": "carnivora",
                "taxon_family": "canidae",
                "taxon_genus": "urocyon",
            },
        },
    )

    vectors = np.zeros((11, 4), dtype=np.float32)
    index = _FakeIndex(nbrs)
    _, top = _compute_neighbor_signals(
        index, vectors, labels, "/dummy.db", "proj"
    )

    assert top[0] is None
