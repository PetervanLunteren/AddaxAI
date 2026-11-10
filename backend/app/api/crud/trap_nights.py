"""
Subfolder-aware trap-nights calculation.

Survey effort for a camera-trap deployment is the inclusive count of days
the camera(s) were deployed. For a clean single-subfolder deployment that's
`max(capture) - min(capture) + 1`. For any other shape the deployment may
contain several subfolders, each with its own capture range; each
subfolder is an independent interval, except that a camera's rollover
folders are joined back into one session.

The algorithm:

1. Bucket files by their parent directory. Each subfolder becomes an
   inclusive interval `[min_capture, max_capture]`.
2. Apply the optional clip window to each interval.
3. Within one parent folder, sorted by start, join an interval to the
   running one when its start equals that one's end. This is the
   Reconyx / Bushnell rollover: `100MEDIA` ends on Jan 15 and `101MEDIA`
   starts on Jan 15, one camera session, so the shared day counts once.
   Per parent folder because a season of cameras bundled as one
   deployment (`cam_01/100RECNX`, `cam_02/100RECNX`, ...) swaps every
   card on the same day, and only a camera's own folders roll over into
   each other.
4. Sum `(end - start).days + 1` across the resulting intervals.

Worked cases (one deployment):

- Single subfolder, Jan 1 - Jan 31 → 31.
- Two sequential SDs with rollover boundary (100MEDIA Jan 1 - Jan 15,
  101MEDIA Jan 15 - Jan 31) → one session, 31.
- Two SDs with a clear gap (Jan 1 - Jan 15 and Feb 1 - Feb 15) → 15 + 15
  = 30.
- Ten parallel cameras bundled as one deployment, each capturing Jan 1 -
  Mar 31 → 10 × 90 = 900. This is the motivating case: the prior
  merge-overlapping algorithm silently collapsed them into 90. This is
  the unpaired reading of subfolders; see "Paired cameras" below.

Until 2026-10 step 3 was a subtraction instead: one day per ordered pair
of subfolders anywhere in the deployment where one's end equalled
another's start. Fifty cameras bundled in one deployment with every card
swapped on the same day matched 50 × 50 pairs per swap and the total went
below zero (shown as 1 on a 923K-file dashboard, every rate 100 × the
count). The join cannot go below zero and has no pairs to overcount.

Paired cameras (`Deployment.paired_cameras`): the subfolders are
dependent cameras (one animal triggers more than one), so the deployment
is one unit of effort.
All its files go in one bucket, giving one interval from the first to
the last capture, and the effort counts once. Two cameras for Jan 1 -
Mar 31 → 90, not 180. There is nothing to join.

Known edge cases the join does not solve:

- **Duplicate-folder accident**: the same SD card imported twice into one
  deployment produces two identical intervals with no shared boundary
  day, so the count doubles. Users usually notice because the file count
  doubles too; a post-hoc check could be added but is out of scope here.
- **Genuine partial multi-day overlap**: e.g. SD1 Jan 1 - Jan 20 and SD2
  Jan 15 - Feb 5. We interpret this as two parallel cameras (sum = 42),
  which is correct for the "bundled backlog" scenario but overcounts the
  rare "one camera with overlapping timestamps due to a clock bug" case.
- **Files in the deployment root next to rollover subfolders**: the
  root's parent is another folder, so the root interval never joins a
  subfolder's. No camera writes that layout (they write `DCIM/100RECNX`).

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

from app.db.sql_params import iter_id_chunks
from app.models import Deployment, File


def _paired_deployment_ids(db: Session, deployment_ids: list[str]) -> set[str]:
    """Ids of the given deployments that have `paired_cameras` on."""
    paired: set[str] = set()
    for chunk in iter_id_chunks(deployment_ids):
        paired.update(
            db.execute(
                select(Deployment.id)
                .where(Deployment.id.in_(chunk))
                .where(Deployment.paired_cameras.is_(True))
            ).scalars()
        )
    return paired


def _bucket_dates(
    rows: list[tuple[str, date]], *, paired_cameras: bool
) -> dict[str, list[date]]:
    """Capture dates per subfolder, or all in one bucket for paired cameras."""
    by_folder: dict[str, list[date]] = defaultdict(list)
    for file_path, captured_at in rows:
        key = "" if paired_cameras else str(Path(file_path).parent)
        by_folder[key].append(captured_at.date())
    return by_folder


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


def _chain_rollovers(
    intervals: list[tuple[date, date]],
) -> list[tuple[date, date]]:
    """Join the rollover chains among one parent folder's intervals: sorted
    by start, an interval whose start equals the running end continues the
    same camera session. Parallel subfolders (same start, or a multi-day
    overlap) do not match and stay separate. Shared with the deployment
    timeline, which draws one bar per session."""
    out: list[tuple[date, date]] = []
    for start, end in sorted(intervals):
        if out and start == out[-1][1]:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def _trap_nights_from_intervals(intervals: list[tuple[date, date]]) -> int:
    return sum((end - start).days + 1 for start, end in intervals)


def _intervals_from_folder_dates(
    by_folder: dict[str, list[date]],
    clip_start: date | None,
    clip_end: date | None,
) -> list[tuple[date, date]]:
    """Camera-session intervals, clipped and sorted by start date: one per
    subfolder, with each parent folder's rollover chains joined. Two
    overlapping subfolders stay as two intervals."""
    by_parent: dict[str, list[tuple[date, date]]] = defaultdict(list)
    for folder, dates in by_folder.items():
        iv = _clip_interval(dates, clip_start, clip_end)
        if iv is not None:
            by_parent[str(Path(folder).parent)].append(iv)
    return sorted(
        iv for ivs in by_parent.values() for iv in _chain_rollovers(ivs)
    )


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
    Otherwise returns the sum of inclusive day counts over the camera
    sessions — see the module docstring for the algorithm.
    """
    rows = db.execute(
        select(File.file_path, File.captured_at_local)
        .where(File.deployment_id == deployment_id)
        .where(File.file_type.in_(("image", "video")))
        .where(File.captured_at_local.isnot(None))
    ).all()

    if not rows:
        return None

    paired = deployment_id in _paired_deployment_ids(db, [deployment_id])
    by_folder = _bucket_dates(rows, paired_cameras=paired)
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
    Camera-session intervals for many deployments in one query.

    Shared primitive for trap-nights accounting and for the deployment
    timeline view, so a bar on the timeline and a night on the dashboard
    are the same thing. Returns `{deployment_id: [(start, end), ...]}`
    with an empty list for deployments that have no capture-bearing files
    within the clip window. Intervals are inclusive `[start, end]`, sorted
    by start date; a camera's rollover folders are one interval, two
    parallel subfolders within one deployment are two. A paired-cameras
    deployment is one interval for the whole deployment.

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

    rows_by_dep: dict[str, list[tuple[str, date]]] = defaultdict(list)
    for deployment_id, file_path, captured_at in rows:
        rows_by_dep[deployment_id].append((file_path, captured_at))
    paired_ids = _paired_deployment_ids(db, list(rows_by_dep))

    result: dict[str, list[tuple[date, date]]] = {
        dep_id: [] for dep_id in deployment_ids
    }
    for dep_id, dep_rows in rows_by_dep.items():
        by_folder = _bucket_dates(dep_rows, paired_cameras=dep_id in paired_ids)
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
