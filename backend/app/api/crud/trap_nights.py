"""
Subfolder-aware trap-nights calculation.

Survey effort for a camera-trap deployment is the inclusive count of days
the camera(s) were deployed. For a clean single-subfolder deployment that's
`max(capture) - min(capture) + 1`. For any other shape the deployment may
contain several subfolders, each with its own capture range; we treat each
subfolder as an independent interval and sum them, with a small correction
for camera-manufacturer rollovers.

The algorithm:

1. Bucket files by their parent directory. Each subfolder becomes an
   inclusive interval `[min_capture, max_capture]`.
2. Apply the optional clip window to each interval.
3. Sum `(end - start).days + 1` across the surviving intervals.
4. Subtract 1 for every ordered pair of subfolders within the same
   deployment where one's end date equals another's start date. This
   collapses the Reconyx / Bushnell rollover case where `100MEDIA` ends on
   Jan 15 and `101MEDIA` starts on Jan 15 so the shared day is counted
   once, not twice.

Worked cases (one deployment):

- Single subfolder, Jan 1 - Jan 31 → 31.
- Two sequential SDs with rollover boundary (100MEDIA Jan 1 - Jan 15,
  101MEDIA Jan 15 - Jan 31) → 15 + 17 - 1 = 31.
- Two SDs with a clear gap (Jan 1 - Jan 15 and Feb 1 - Feb 15) → 15 + 15
  = 30.
- Ten parallel cameras bundled as one deployment, each capturing Jan 1 -
  Mar 31 → 10 × 90 = 900. This is the motivating fix: the prior
  merge-overlapping algorithm silently collapsed them into 90.

Known edge cases the boundary-subtraction rule does not fully solve:

- **Duplicate-folder accident**: the same SD card imported twice into one
  deployment produces two identical intervals with no shared boundary
  day, so the count doubles. Users usually notice because the file count
  doubles too; a post-hoc check could be added but is out of scope here.
- **Genuine partial multi-day overlap**: e.g. SD1 Jan 1 - Jan 20 and SD2
  Jan 15 - Feb 5. We interpret this as two parallel cameras (sum = 42),
  which is correct for the "bundled backlog" scenario but overcounts the
  rare "one camera with overlapping timestamps due to a clock bug" case.
- **Triple boundary at one day**: three subfolders with pairwise
  end↔start matches on one date over-subtract by 1 (counts the shared
  day zero times instead of once). Rare; off by one day.

Frame rows (`file_type='frame'`) are pipeline artifacts living inside
`.addaxai/video_frames/...`; they are filtered out. Each original capture
is represented by an `image` or `video` row.
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
    """Inclusive [min, max] for a subfolder's dates, clamped to the clip
    window. Returns None if the subfolder has no dates or the clip window
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


def _count_boundary_matches(intervals: list[tuple[date, date]]) -> int:
    """Number of ordered pairs (i, j) with i != j where `intervals[i].end`
    equals `intervals[j].start`. Each match corresponds to one shared
    boundary day that would otherwise be double-counted when summing
    per-subfolder spans."""
    count = 0
    for i, (_, end_i) in enumerate(intervals):
        for j, (start_j, _) in enumerate(intervals):
            if i != j and end_i == start_j:
                count += 1
    return count


def _trap_nights_from_intervals(intervals: list[tuple[date, date]]) -> int:
    """Sum subfolder spans, then subtract boundary matches."""
    total = sum((end - start).days + 1 for start, end in intervals)
    return total - _count_boundary_matches(intervals)


def _intervals_from_folder_dates(
    by_folder: dict[str, list[date]],
    clip_start: date | None,
    clip_end: date | None,
) -> list[tuple[date, date]]:
    """Per-subfolder intervals, clipped and sorted by start date.
    No merging: two overlapping subfolders stay as two intervals."""
    intervals = [
        iv
        for dates in by_folder.values()
        if (iv := _clip_interval(dates, clip_start, clip_end)) is not None
    ]
    return sorted(intervals)


def compute_trap_nights_for_deployment(
    db: Session,
    deployment_id: str,
    *,
    clip_start: date | None = None,
    clip_end: date | None = None,
) -> int | None:
    """
    Subfolder-aware trap-nights count for a single deployment.

    Returns `None` when the deployment has no capture-bearing file rows.
    Returns `0` when files exist but the clip window excludes all of them.
    Otherwise returns the sum of per-subfolder inclusive day counts, minus
    1 per shared boundary day — see the module docstring for the algorithm.
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

    intervals = _intervals_from_folder_dates(by_folder, clip_start, clip_end)
    return _trap_nights_from_intervals(intervals)


def compute_intervals_for_deployments(
    db: Session,
    deployment_ids: list[str],
    *,
    clip_start: date | None = None,
    clip_end: date | None = None,
) -> dict[str, list[tuple[date, date]]]:
    """
    Per-subfolder capture intervals for many deployments in one query.

    Shared primitive for trap-nights accounting and for the deployment
    timeline view. Returns `{deployment_id: [(start, end), ...]}` with an
    empty list for deployments that have no capture-bearing files within
    the clip window. Intervals are inclusive `[start, end]`, sorted by
    start date, and **not merged**: two parallel subfolders within one
    deployment produce two separate intervals.

    See the module docstring for the algorithm.
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

    result: dict[str, list[tuple[date, date]]] = {
        dep_id: [] for dep_id in deployment_ids
    }
    for dep_id, by_folder in by_dep_folder.items():
        result[dep_id] = _intervals_from_folder_dates(
            by_folder, clip_start, clip_end
        )
    return result


def compute_trap_nights_for_deployments(
    db: Session,
    deployment_ids: list[str],
    *,
    clip_start: date | None = None,
    clip_end: date | None = None,
) -> dict[str, int]:
    """
    Subfolder-aware trap-nights count for many deployments in one query.

    Used by dashboard-style aggregations where iterating per deployment
    would amplify the query cost. Returns `{deployment_id: nights}` with
    `0` for deployments that have no capture-bearing files (not `None` —
    bulk callers divide and want a numeric zero, not a missing key).

    Thin wrapper around `compute_intervals_for_deployments`: single source
    of truth for the subfolder-bucketing logic.
    """
    intervals = compute_intervals_for_deployments(
        db, deployment_ids, clip_start=clip_start, clip_end=clip_end
    )
    return {
        dep_id: _trap_nights_from_intervals(ivs)
        for dep_id, ivs in intervals.items()
    }
