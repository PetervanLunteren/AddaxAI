"""
Pure-Python ordering helpers for the Observations sort endpoint.

The similarity walk itself runs inside the ML subprocess (numpy + FAISS).
This module owns the deterministic part: which order to present results
in once metadata is in hand. Kept stdlib-only so the main backend can
import and unit-test it.
"""

from __future__ import annotations

VALID_SORTS: frozenset[str] = frozenset(
    {
        "similarity",
        "similarity_reverse",
        "newest",
        "oldest",
        "cls_low",
        "suggestions",
    }
)


def order_indices(
    sort_mode: str,
    similarity_order: list[int],
    metas: list[dict],
) -> list[int]:
    """Pick the final index order for the requested sort mode.

    `similarity_order` is the greedy-walk order; it is returned as-is for
    `similarity` and reversed for `similarity_reverse`. The metadata-based
    sorts ignore embeddings and order by `captured_at_local` or
    `label_confidence`. NULL values (missing timestamp or unscored
    detection) sort to the end so they don't dominate the head of the
    grid.

    `suggestions` is intentionally NOT handled here. It needs the
    neighbour signals (`top_labels`, `agreement_scores`) that this
    module doesn't see; do_sort branches on it and calls
    `suggestions_order` directly.
    """
    if sort_mode not in VALID_SORTS:
        raise ValueError(f"Unknown sort mode: {sort_mode}")
    if sort_mode == "suggestions":
        raise ValueError(
            "sort_mode='suggestions' is handled by do_sort, not order_indices"
        )

    n = len(metas)
    if sort_mode == "similarity":
        return list(similarity_order)
    if sort_mode == "similarity_reverse":
        return list(reversed(similarity_order))

    if sort_mode in ("newest", "oldest"):
        descending = sort_mode == "newest"
        with_ts = [(metas[i].get("captured_at_local"), i) for i in range(n)]
        non_null = [(ts, i) for ts, i in with_ts if ts]
        nulls = [i for ts, i in with_ts if not ts]
        non_null.sort(key=lambda kv: kv[0], reverse=descending)
        return [i for _, i in non_null] + nulls

    # cls_low
    with_lc = [(metas[i].get("label_confidence"), i) for i in range(n)]
    non_null = [(lc, i) for lc, i in with_lc if lc is not None]
    nulls = [i for lc, i in with_lc if lc is None]
    non_null.sort(key=lambda kv: kv[0])
    return [i for _, i in non_null] + nulls


def suggestions_order(
    metas: list[dict],
    top_labels: list[str | None],
    agreement_scores: list[float],
    *,
    min_count: int = 8,
    max_cohorts: int = 200,
) -> list[int]:
    """Cohort-grouped order for the suggestions sort mode.

    Filters to unverified detections that carry a descendant-promotion
    suggestion (`top_labels[i]` is set), groups them by
    `(label, top_label, category)`, drops cohorts with fewer than
    `min_count` members, keeps the top `max_cohorts` by count, then
    orders cohort by cohort: groups by descending member count (with a
    deterministic tiebreaker), within each group by ascending neighbour
    agreement so the strongest promotion candidates lead.

    The `min_count` / `max_cohorts` defaults mirror the cohorts endpoint
    so the toolbar's count signal matches the grid's content exactly.

    Stdlib-only: takes plain Python lists so the main backend can
    unit-test it without numpy / FAISS.
    """
    if not (len(metas) == len(top_labels) == len(agreement_scores)):
        raise ValueError(
            "metas, top_labels and agreement_scores must be the same length"
        )
    if min_count < 1 or max_cohorts < 1:
        raise ValueError(
            f"min_count and max_cohorts must be >= 1 "
            f"(got min_count={min_count}, max_cohorts={max_cohorts})"
        )

    # Bucket eligible detections by cohort key.
    buckets: dict[tuple[str, str, str], list[tuple[float, int]]] = {}
    for i, meta in enumerate(metas):
        if meta.get("verified"):
            continue
        # User dismissed this crop's suggestion: keep it as a neighbour
        # vote (it's still in metas) but never make it a cohort member.
        if meta.get("suggestion_dismissed"):
            continue
        suggested = top_labels[i]
        if not suggested:
            continue
        key = (meta.get("label") or "", suggested, meta.get("category") or "")
        buckets.setdefault(key, []).append((agreement_scores[i], i))

    # Drop small cohorts, sort by descending count (deterministic tiebreaker
    # on key for stable ordering across reruns), keep the top N.
    qualified = [
        (key, members) for key, members in buckets.items() if len(members) >= min_count
    ]
    qualified.sort(key=lambda item: (-len(item[1]), item[0]))
    qualified = qualified[:max_cohorts]

    # Sort each cohort by ascending agreement (most-disagreeing first).
    for _, members in qualified:
        members.sort(key=lambda pair: pair[0])

    return [i for _, members in qualified for _, i in members]
