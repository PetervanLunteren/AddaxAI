"""
Pure-Python ordering helpers for the Observations sort endpoint.

The similarity walk itself runs inside the ML subprocess (numpy + FAISS).
This module owns the deterministic part: which order to present results
in once metadata is in hand. Kept stdlib-only so the main backend can
import and unit-test it.
"""

from __future__ import annotations

VALID_SORTS: frozenset[str] = frozenset(
    {"similarity", "similarity_reverse", "newest", "oldest", "cls_low"}
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
    """
    if sort_mode not in VALID_SORTS:
        raise ValueError(f"Unknown sort mode: {sort_mode}")

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
