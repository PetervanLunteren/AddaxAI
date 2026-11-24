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

import os
from collections import Counter, defaultdict
from datetime import date, timedelta

from sqlalchemy import String, func, select
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


def _folder_spans_by_deployment(
    db: Session, deployment_ids: list[str]
) -> dict[str, dict[str, tuple[date, date]]]:
    """First and last capture day of every subfolder, per deployment.

    `{deployment_id: {folder: (first_day, last_day)}}`, with one bucket
    `""` for a paired-cameras deployment. Deployments without
    capture-bearing files are absent.

    This reads every capture file of the deployments, so it is the cost of
    trap nights on a big project. Two choices keep it cheap:

    - The day is read as text, the first 10 characters of
      `captured_at_local` (`2025-01-15 23:59:59.5` gives `2025-01-15`).
      ISO day strings compare like dates, so first and last are kept as
      text and only those become `date`s. Building a datetime per row was
      half the time: 1.27M files took 3.4 s, now 1.7 s.
    - `os.path.dirname`, not `Path(...).parent`: on 480,000 files pathlib
      took 3.3 s of a 7.5 s request where dirname takes 0.13 s. The keys
      are only compared with each other.
    - The select runs on the session's connection, in the same
      transaction. `db.execute` would send these 1.27M rows through the
      ORM's result handling, which wraps nothing here and cost 0.8 s.
    """
    rows = db.connection().execute(
        select(
            File.deployment_id,
            File.file_path,
            func.substr(File.captured_at_local, 1, 10, type_=String),
        )
        .where(File.deployment_id.in_(deployment_ids))
        .where(File.file_type.in_(("image", "video")))
        .where(File.captured_at_local.isnot(None))
    )
    paired_ids = _paired_deployment_ids(db, deployment_ids)
    spans: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    dirname = os.path.dirname
    for deployment_id, file_path, day in rows:
        folders = spans[deployment_id]
        key = "" if deployment_id in paired_ids else dirname(file_path)
        span = folders.get(key)
        if span is None:
            folders[key] = (day, day)
        elif day < span[0]:
            folders[key] = (day, span[1])
        elif day > span[1]:
            folders[key] = (span[0], day)
    return {
        deployment_id: {
            folder: (date.fromisoformat(first), date.fromisoformat(last))
            for folder, (first, last) in folders.items()
        }
        for deployment_id, folders in spans.items()
    }


def _clip_interval(
    span: tuple[date, date],
    clip_start: date | None,
    clip_end: date | None,
) -> tuple[date, date] | None:
    """A subfolder's inclusive [first, last] day, clamped to the clip
    window. Returns None if the clip window excludes everything."""
    mn, mx = span
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


def nights_per_day(intervals: list[tuple[date, date]]) -> Counter[date]:
    """How many camera sessions ran on each day: the same effort as
    `_trap_nights_from_intervals`, spread over the calendar. Two parallel
    cameras on one day are 2 trap nights on 1 active day. The sum of the
    counts is the trap nights, so effort per period (the trend charts)
    and effort in total (every other rate) cannot disagree."""
    nights: Counter[date] = Counter()
    for start, end in intervals:
        for offset in range((end - start).days + 1):
            nights[start + timedelta(days=offset)] += 1
    return nights


def _intervals_from_folder_spans(
    by_folder: dict[str, tuple[date, date]],
    clip_start: date | None,
    clip_end: date | None,
) -> list[tuple[date, date]]:
    """Camera-session intervals, clipped and sorted by start date: one per
    subfolder, with each parent folder's rollover chains joined. Two
    overlapping subfolders stay as two intervals."""
    by_parent: dict[str, list[tuple[date, date]]] = defaultdict(list)
    for folder, span in by_folder.items():
        iv = _clip_interval(span, clip_start, clip_end)
        if iv is not None:
            by_parent[os.path.dirname(folder)].append(iv)
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
    by_folder = _folder_spans_by_deployment(db, [deployment_id]).get(deployment_id)
    if not by_folder:
        return None
    intervals = _intervals_from_folder_spans(by_folder, clip_start, clip_end)
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

    result: dict[str, list[tuple[date, date]]] = {
        dep_id: [] for dep_id in deployment_ids
    }
    for dep_id, by_folder in _folder_spans_by_deployment(db, deployment_ids).items():
        result[dep_id] = _intervals_from_folder_spans(by_folder, clip_start, clip_end)
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
