"""
Deployment timeline CRUD.

Builds the payload for the Insights → Deployment timeline view. Reuses the
folder-aware interval primitive from `trap_nights.py` so the bars, the
concurrent-cameras sweep, and the Dashboard's trap-nights total all read
the same source of truth.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.crud.deployment import site_ids_filter
from app.api.crud.trap_nights import compute_intervals_for_deployments
from app.api.schemas.timeline import (
    ConcurrentPoint,
    TimelineDeployment,
    TimelineMetrics,
    TimelineResponse,
    TimelineSite,
    TrapNightInterval,
)
from app.models import Deployment, File, Site

NO_SITE_LABEL = "(no site)"


def _collapse_rollover_chains(
    intervals: list[tuple[date, date]],
) -> list[tuple[date, date]]:
    """Merge adjacent intervals within a single deployment that share an
    exact boundary day (one's `end` equals the next one's `start`). This
    is the Reconyx / Bushnell rollover case: `100MEDIA.end` == `101MEDIA.start`
    is one camera session, not two. Used for both the Gantt bars and the
    concurrent-cameras sweep, so the bar + tooltip show the whole chain
    as one continuous interval and the sweep does not spike on the shared
    day. Genuinely parallel subfolders (same start and same end, or
    multi-day partial overlap) do NOT match the rule and stay separate.

    This is a timeline-presentation concern. `trap_nights.py` still
    returns one interval per subfolder; the collapse happens here."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = out[-1]
        if start == last_end:
            out[-1] = (last_start, end)
        else:
            out.append((start, end))
    return out


def _concurrent_sweep(
    intervals: list[tuple[date, date]],
) -> list[ConcurrentPoint]:
    """Sweep-line over inclusive intervals, emit change points only.

    For each interval `[start, end]` emit `(start, +1)` and
    `(end + 1, -1)` (half-open on the right), sort by date, accumulate.
    Emit one `ConcurrentPoint` per distinct date where the running count
    changes, skipping no-op duplicates. The final point always drops to
    zero so the frontend can close the area chart cleanly.
    """
    if not intervals:
        return []
    events: list[tuple[date, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end + timedelta(days=1), -1))
    events.sort(key=lambda e: e[0])

    points: list[ConcurrentPoint] = []
    running = 0
    i = 0
    n = len(events)
    while i < n:
        day = events[i][0]
        # Collapse multiple events on the same day into one delta so the
        # emitted series has at most one point per calendar day.
        delta = 0
        while i < n and events[i][0] == day:
            delta += events[i][1]
            i += 1
        running += delta
        points.append(ConcurrentPoint(date=day, count=running))
    return points


def _deployment_label(folder_path: str | None, deployment_id: str) -> str:
    if folder_path:
        name = Path(folder_path).name
        if name:
            return name
    return deployment_id[:8]


def get_deployment_timeline(
    db: Session,
    project_id: str,
    *,
    site_ids: list[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> TimelineResponse:
    """Build the Insights → Deployment timeline payload for a project."""
    query = select(Deployment).where(Deployment.project_id == project_id)
    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.where(site_clause)
    deployments = list(db.execute(query).scalars().all())

    if not deployments:
        return TimelineResponse(
            sites=[],
            concurrent_cameras=[],
            metrics=TimelineMetrics(
                site_count=0,
                deployment_count=0,
                total_trap_nights=0,
                median_deployment_length_days=None,
                max_concurrent_cameras=0,
            ),
            date_range_from=None,
            date_range_to=None,
        )

    deployment_ids = [d.id for d in deployments]
    intervals_by_dep = compute_intervals_for_deployments(
        db, deployment_ids, clip_start=date_from, clip_end=date_to
    )

    # One query for file counts per deployment (tooltip metric).
    file_counts_rows = db.execute(
        select(File.deployment_id, func.count(File.id))
        .where(File.deployment_id.in_(deployment_ids))
        .where(File.file_type.in_(("image", "video")))
        .group_by(File.deployment_id)
    ).all()
    file_counts: dict[str, int] = {dep_id: count for dep_id, count in file_counts_rows}

    # One query for site names.
    site_ids_needed = {d.site_id for d in deployments if d.site_id is not None}
    site_names: dict[str, str] = {}
    if site_ids_needed:
        site_rows = db.execute(
            select(Site.id, Site.name).where(Site.id.in_(site_ids_needed))
        ).all()
        site_names = {sid: name for sid, name in site_rows}

    # Group deployments by site_id and build the response rows. A
    # deployment with zero intervals (no files in the clip window) still
    # renders its configured outer bar so users can see the empty slot.
    by_site: dict[str | None, list[TimelineDeployment]] = defaultdict(list)
    interval_lengths: list[int] = []
    all_intervals: list[tuple[date, date]] = []

    for dep in deployments:
        # Collapse rollover chains up-front so the Gantt bars, tooltips,
        # sweep, and metrics all read the same continuous-capture intervals.
        # Per-subfolder granularity is abandoned for rollover chains (one
        # camera session, one bar). Parallel / multi-day-overlap subfolders
        # stay separate.
        dep_intervals = _collapse_rollover_chains(
            intervals_by_dep.get(dep.id, [])
        )
        tn_intervals = [
            TrapNightInterval(
                start=s, end=e, trap_nights=(e - s).days + 1
            )
            for s, e in dep_intervals
        ]
        for s, e in dep_intervals:
            interval_lengths.append((e - s).days + 1)
        all_intervals.extend(dep_intervals)

        by_site[dep.site_id].append(
            TimelineDeployment(
                deployment_id=dep.id,
                deployment_label=_deployment_label(dep.folder_path, dep.id),
                camera_model=dep.camera_model,
                configured_start=dep.start_date_local,
                configured_end=dep.end_date_local,
                intervals=tn_intervals,
                file_count=file_counts.get(dep.id, 0),
            )
        )

    # Sort deployments inside each site by configured_start for stable rendering.
    sites: list[TimelineSite] = []
    real_site_entries = [
        (sid, name)
        for sid, name in site_names.items()
        if sid in by_site
    ]
    real_site_entries.sort(key=lambda e: e[1].lower())
    for sid, name in real_site_entries:
        deps = sorted(by_site[sid], key=lambda d: d.configured_start)
        sites.append(TimelineSite(site_id=sid, site_name=name, deployments=deps))
    if None in by_site:
        deps = sorted(by_site[None], key=lambda d: d.configured_start)
        sites.append(TimelineSite(site_id=None, site_name=NO_SITE_LABEL, deployments=deps))

    concurrent = _concurrent_sweep(all_intervals)

    total_trap_nights = sum(
        iv.trap_nights
        for s in sites
        for d in s.deployments
        for iv in d.intervals
    )
    median_len = median(interval_lengths) if interval_lengths else None
    max_concurrent = max((p.count for p in concurrent), default=0)

    # Observed x-axis span: earliest configured start to latest of
    # (configured_end, last observed interval end). Honours clip window.
    observed_starts = [d.configured_start for s in sites for d in s.deployments]
    observed_ends: list[date] = []
    for s in sites:
        for d in s.deployments:
            if d.configured_end is not None:
                observed_ends.append(d.configured_end)
            for iv in d.intervals:
                observed_ends.append(iv.end)
    date_range_from = min(observed_starts) if observed_starts else None
    date_range_to = max(observed_ends) if observed_ends else None
    if date_from is not None and date_range_from is not None:
        date_range_from = max(date_range_from, date_from)
    if date_to is not None and date_range_to is not None:
        date_range_to = min(date_range_to, date_to)

    return TimelineResponse(
        sites=sites,
        concurrent_cameras=concurrent,
        metrics=TimelineMetrics(
            site_count=len([s for s in sites if s.site_id is not None]),
            deployment_count=sum(len(s.deployments) for s in sites),
            total_trap_nights=total_trap_nights,
            median_deployment_length_days=(
                float(median_len) if median_len is not None else None
            ),
            max_concurrent_cameras=max_concurrent,
        ),
        date_range_from=date_range_from,
        date_range_to=date_range_to,
    )
