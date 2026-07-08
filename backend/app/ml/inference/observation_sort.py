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
        "events",
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

    if sort_mode == "events":
        # Group detections by their event. Events are ordered newest
        # first (matching the Counts page default); within an event,
        # chronological by sequence_number so a burst reads in capture
        # order. Detections with no event (event clustering not run, or
        # an orphaned file) sort to the end, like the timestamp sorts.
        #
        # The sort is stable, so we sort by the weakest key first
        # (sequence ascending) then the strongest (event_start
        # descending). event_id breaks ties between events that share a
        # start time so their detections never interleave.
        with_event = [(metas[i].get("event_id"), i) for i in range(n)]
        non_null = [i for eid, i in with_event if eid]
        nulls = [i for eid, i in with_event if not eid]
        non_null.sort(key=lambda i: (metas[i].get("event_sequence") or 0))
        non_null.sort(
            key=lambda i: (
                metas[i].get("event_start_local") or "",
                metas[i].get("event_id") or "",
            ),
            reverse=True,
        )
        return non_null + nulls

    # cls_low
    with_lc = [(metas[i].get("label_confidence"), i) for i in range(n)]
    non_null = [(lc, i) for lc, i in with_lc if lc is not None]
    nulls = [i for lc, i in with_lc if lc is None]
    non_null.sort(key=lambda kv: kv[0])
    return [i for _, i in non_null] + nulls


def order_events_by_deployment(metas: list[dict]) -> list[int]:
    """Deployment-grouped chronological order for the no-embedding event
    sort. Keeps each camera's (deployment's) events together instead of
    interleaving cameras by time, so a multi-camera project reviews one
    camera at a time.

    Cameras are ordered by their most recent event; within a camera,
    events are newest-first; within an event, by capture sequence.
    Detections with no event sort to the end. For a single-deployment
    folder run every detection shares one deployment, so this reduces
    exactly to the plain chronological ``order_indices("events", ...)``.

    Used by the no-embedding event sort and as the baseline for the
    similarity event sort, so the embedless tail (events with no
    embedded detection) is camera-grouped in both. Events that do have
    a representative are reordered by appearance on top of this.
    """
    n = len(metas)

    # Each deployment's newest event start = its recency for camera order.
    dep_newest: dict[str, str] = {}
    for m in metas:
        dep = m.get("deployment_id")
        if dep is None:
            continue
        start = m.get("event_start_local") or ""
        if dep not in dep_newest or start > dep_newest[dep]:
            dep_newest[dep] = start

    with_event = [i for i in range(n) if metas[i].get("event_id")]
    no_event = [i for i in range(n) if not metas[i].get("event_id")]

    # Stable multi-pass, weakest key first: sequence within event, then
    # event (newest first), then camera (newest activity first).
    with_event.sort(key=lambda i: metas[i].get("event_sequence") or 0)
    with_event.sort(
        key=lambda i: (
            metas[i].get("event_start_local") or "",
            metas[i].get("event_id") or "",
        ),
        reverse=True,
    )
    with_event.sort(
        key=lambda i: (
            dep_newest.get(metas[i].get("deployment_id"), ""),
            metas[i].get("deployment_id") or "",
        ),
        reverse=True,
    )
    return with_event + no_event


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
