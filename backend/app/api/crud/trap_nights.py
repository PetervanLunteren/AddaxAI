"""
Folder-aware trap-nights calculation.

Survey effort for a camera-trap deployment is the inclusive count of days
the camera was deployed. For a clean single-folder deployment that's
`max(capture) - min(capture) + 1`. For a mixed backlog (one AddaxAI
deployment row wrapping several SD-card folders spaced apart in time), that
naive span includes the offline gaps between cards and wildly inflates the
denominator of any `observations / trap_nights * 100` rate.

The algorithm:

1. Bucket files by their parent directory. Each folder becomes a date
   interval `[min_capture, max_capture]`, inclusive.
2. Merge overlapping intervals within a deployment, then sum
   `(end - start).days + 1` across the merged result.

For a clean single-folder deployment this equals `max - min + 1` — one
folder, one interval, no merging. For a backlog with three disjoint SD
cards it equals the sum of each card's run length (intervals don't
overlap so merging is a no-op). For a camera-manufacturer rollover where
two adjacent folders share a boundary day (`100MEDIA` ending on Jan 15
and `101MEDIA` starting on Jan 15), the overlap causes the shared day to
collapse so the count stays 30 rather than 31.

Frame rows (`file_type='frame'`) are pipeline artifacts living inside
`.addaxai/video_frames/...` — they are filtered out. Each original
capture is already represented by an `image` or `video` row.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import File


def _clip_interval(
    dates: list[date],
    clip_start: date | None,
    clip_end: date | None,
) -> tuple[date, date] | None:
    """Inclusive [min, max] for a folder's dates, clamped to the clip
    window. Returns None if the folder has no dates or the clip window
    excludes everything."""
    if not dates:
        return None
    mn = min(dates)
    mx = max(dates)
    if clip_start is not None:
        mn = max(mn, clip_start)
    if clip_end is not None:
        mx = min(mx, clip_end)
    if mx < mn:
        return None
    return mn, mx


def _merge_overlapping(
    intervals: list[tuple[date, date]]
) -> list[tuple[date, date]]:
    """Merge intervals that share any day. Two intervals `[a, b]` and
    `[c, d]` with `a <= c` are considered overlapping when `c <= b`.
    Adjacent intervals that don't share a day (e.g. `[a, b]` then
    `[b+1, d]`) are left separate — adjacency doesn't affect the total
    inclusive day count."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[date, date]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _sum_intervals(intervals: list[tuple[date, date]]) -> int:
    """Inclusive day count across a set of already-merged intervals."""
    return sum((end - start).days + 1 for start, end in intervals)


def _trap_nights_from_folder_dates(
    by_folder: dict[str, list[date]],
    clip_start: date | None,
    clip_end: date | None,
) -> int:
    """Shared kernel: per-folder intervals, merge overlaps, sum."""
    intervals = [
        interval
        for dates in by_folder.values()
        if (interval := _clip_interval(dates, clip_start, clip_end)) is not None
    ]
    return _sum_intervals(_merge_overlapping(intervals))


def compute_trap_nights_for_deployment(
    db: Session,
    deployment_id: str,
    *,
    clip_start: date | None = None,
    clip_end: date | None = None,
) -> int | None:
    """
    Folder-aware trap-nights count for a single deployment.

    Returns `None` when the deployment has no capture-bearing file rows.
    Returns `0` when files exist but the clip window excludes all of them.
    Otherwise returns the inclusive day count of the merged per-folder
    intervals — see the module docstring for the algorithm.
    """
    rows = db.execute(
        select(File.file_path, File.captured_at_local)
        .where(File.deployment_id == deployment_id)
        .where(File.file_type.in_(("image", "video")))
        .where(File.captured_at_local.isnot(None))
    ).all()

    if not rows:
        return None

    by_folder: dict[str, list[date]] = defaultdict(list)
    for file_path, captured_at in rows:
        folder = str(Path(file_path).parent)
        by_folder[folder].append(captured_at.date())

    return _trap_nights_from_folder_dates(by_folder, clip_start, clip_end)


def compute_trap_nights_for_deployments(
    db: Session,
    deployment_ids: list[str],
    *,
    clip_start: date | None = None,
    clip_end: date | None = None,
) -> dict[str, int]:
    """
    Folder-aware trap-nights count for many deployments in one query.

    Used by dashboard-style aggregations where iterating per deployment
    would amplify the query cost. Returns `{deployment_id: nights}` with
    `0` for deployments that have no capture-bearing files (not `None` —
    bulk callers divide and want a numeric zero, not a missing key).
    """
    if not deployment_ids:
        return {}

    rows = db.execute(
        select(File.deployment_id, File.file_path, File.captured_at_local)
        .where(File.deployment_id.in_(deployment_ids))
        .where(File.file_type.in_(("image", "video")))
        .where(File.captured_at_local.isnot(None))
    ).all()

    by_dep_folder: dict[str, dict[str, list[date]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for deployment_id, file_path, captured_at in rows:
        folder = str(Path(file_path).parent)
        by_dep_folder[deployment_id][folder].append(captured_at.date())

    totals: dict[str, int] = {dep_id: 0 for dep_id in deployment_ids}
    for dep_id, by_folder in by_dep_folder.items():
        totals[dep_id] = _trap_nights_from_folder_dates(
            by_folder, clip_start, clip_end
        )
    return totals
