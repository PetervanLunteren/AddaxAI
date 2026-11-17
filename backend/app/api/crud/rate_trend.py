"""Observations over time: the Explore trend card and the Trend page.

The one place that cuts time into periods: days, ISO weeks and calendar
months, for two charts with two jobs.

- **Explore** (`get_detection_trend`): observations per period, a count.
  The dashboard has to be quick, and effort per period means reading
  every file row of the scope (1.6 s on 480,000 files, three times over
  when it runs next to the tiles and the map). A count needs one grouped
  query. Deployments without a site count, like the Explore tiles.
- **Trend page** (`get_rate_trend`): the rate, with its uncertainty. The
  site is the sampling unit. Each site gets its own rate against its
  effort in the period, and the period is the mean of those rates with a
  95% confidence interval (`app.core.rate_statistics`). A site with
  effort and no observations is a rate of 0. Deployments without a site
  have no sampling unit and are left out.

Which sites count:

- In a period of the chart, a site active on at least 80% of the period's
  days (`MIN_ACTIVE_SHARE`), counted over the whole calendar period even
  where the date filter cuts it. Cameras rarely start or stop on a
  period's edge, and 1 observation over 1 trap night would read as 7 per
  week. A period no site ran long enough has no mean at all: a gap, not
  a zero.
- In the species table, every site with at least 1 trap night in the
  date range. Over a whole range a sliver of effort is rare, and a rule
  as strict as the chart's would empty the table for rotating cameras.

Rates leave here per 100 trap nights, like every backend rate. The user's
rate unit is applied by the frontend only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import NamedTuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.crud.deployment import site_ids_filter
from app.api.crud.statistics import _scoped_observations_query
from app.api.crud.trap_nights import (
    compute_intervals_for_deployments,
    nights_per_day,
)
from app.api.schemas.statistics import (
    DetectionTrendPoint,
    DetectionTrendResponse,
    RateTrendResponse,
    SpeciesRate,
    TrendGranularity,
    TrendPeriod,
)
from app.core.rate_statistics import RateStats, summarise_rates
from app.models.deployment import Deployment
from app.models.event import Event
from app.models.event_observation import EventObservation
from app.models.label_taxonomy import LabelTaxonomy

# A site counts in a period of the Trend chart when it was active on at
# least this share of the period's days.
MIN_ACTIVE_SHARE = 0.8

# A span of capture years wider than this is a broken date, not a survey
# (one corrupt EXIF year stretches a project over millennia). Mirrors
# MAX_CHART_YEARS in frontend/src/lib/date-span.ts, which shows the notice.
MAX_CHART_YEARS = 30

# A deployment's site id, or None for a deployment without a site.
SiteKey = str | None
# What the species table has a row for: the raw label (or, without one,
# the detector category) and its taxonomy id.
LabelKey = tuple[str, str | None]


class Period(NamedTuple):
    key: str  # "2024-01-15", "2024-W03" or "2024-01"
    start: date
    end: date  # inclusive


def period_of(day: date, granularity: TrendGranularity) -> Period:
    """The day, ISO week (Monday to Sunday) or calendar month a day is in."""
    if granularity == "day":
        return Period(day.isoformat(), day, day)
    if granularity == "week":
        monday = day - timedelta(days=day.weekday())
        iso = day.isocalendar()
        return Period(
            f"{iso.year:04d}-W{iso.week:02d}", monday, monday + timedelta(days=6)
        )
    first = day.replace(day=1)
    next_first = (first + timedelta(days=32)).replace(day=1)
    return Period(
        f"{day.year:04d}-{day.month:02d}", first, next_first - timedelta(days=1)
    )


def build_periods(
    first: date, last: date, granularity: TrendGranularity
) -> list[Period]:
    """Every period from the one holding `first` to the one holding `last`,
    without gaps."""
    periods = [period_of(first, granularity)]
    while periods[-1].end < last:
        periods.append(period_of(periods[-1].end + timedelta(days=1), granularity))
    return periods


def pick_granularity(first: date, last: date, *, allow_day: bool) -> TrendGranularity:
    """Days for a short survey, weeks for a season, months for years."""
    days = (last - first).days + 1
    if days > 365:
        return "month"
    if days > 90 or not allow_day:
        return "week"
    return "day"


@dataclass
class _LabelNames:
    label: str
    label_taxonomy_id: str | None
    common_name: str
    scientific_name: str


@dataclass
class _ScopeData:
    """Effort and observations of a scope, per site and day."""

    # First and last day a camera ran, or None without any effort.
    first: date | None = None
    last: date | None = None
    # The span of years is too wide to chart; nothing below is filled.
    too_wide: bool = False
    # site -> day -> camera sessions running that day
    nights: dict[SiteKey, Counter[date]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    # (site, day, label) -> observations
    observations: Counter[tuple[SiteKey, date, LabelKey]] = field(
        default_factory=Counter
    )
    names: dict[LabelKey, _LabelNames] = field(default_factory=dict)

    def span(self) -> tuple[date, date] | None:
        """First and last day to chart, or None when there is no effort
        or the years are too wide."""
        if self.first is None or self.last is None or self.too_wide:
            return None
        return self.first, self.last


def _too_wide(first: date, last: date) -> bool:
    return last.year - first.year + 1 > MAX_CHART_YEARS


def _load_scope(
    db: Session,
    project_id: str,
    label_taxonomy_ids: list[str] | None,
    site_ids: list[str] | None,
    date_from: str | None,
    date_to: str | None,
) -> _ScopeData:
    """Effort per site and day, and observations per site, day and label.

    Deployments without a site are left out: the per-site statistics have
    no unit to put them in.
    """
    data = _ScopeData()

    dep_query = (
        select(Deployment.id, Deployment.site_id)
        .where(Deployment.project_id == project_id)
        .where(Deployment.site_id.isnot(None))
    )
    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        dep_query = dep_query.where(site_clause)
    site_by_dep: dict[str, SiteKey] = {
        row.id: row.site_id for row in db.execute(dep_query).all()
    }

    intervals_by_dep = compute_intervals_for_deployments(
        db,
        list(site_by_dep),
        clip_start=date.fromisoformat(date_from) if date_from else None,
        clip_end=date.fromisoformat(date_to) if date_to else None,
    )
    all_intervals = [iv for ivs in intervals_by_dep.values() for iv in ivs]
    if not all_intervals:
        return data
    data.first = min(start for start, _ in all_intervals)
    data.last = max(end for _, end in all_intervals)
    # Checked before the intervals are spread over days: a corrupt year
    # would otherwise be walked a million days at a time.
    if _too_wide(data.first, data.last):
        data.too_wide = True
        return data

    for dep_id, intervals in intervals_by_dep.items():
        data.nights[site_by_dep[dep_id]].update(nights_per_day(intervals))

    day_expr = func.strftime("%Y-%m-%d", Event.event_start_local)
    query = (
        _scoped_observations_query(
            [
                Deployment.site_id,
                day_expr.label("day"),
                EventObservation.label,
                EventObservation.category,
                EventObservation.label_taxonomy_id,
                LabelTaxonomy.common_name,
                LabelTaxonomy.scientific_name,
                func.sum(EventObservation.effective_count).label("count"),
            ],
            project_id, label_taxonomy_ids, site_ids, date_from, date_to,
        )
        .outerjoin(
            LabelTaxonomy,
            LabelTaxonomy.id == EventObservation.label_taxonomy_id,
        )
        # Date-less events (NULL bounds) have no day; exclude them.
        .where(Event.event_start_local.isnot(None))
        .where(Deployment.site_id.isnot(None))
        .group_by(
            Deployment.site_id,
            day_expr,
            EventObservation.label,
            EventObservation.category,
            EventObservation.label_taxonomy_id,
            LabelTaxonomy.common_name,
            LabelTaxonomy.scientific_name,
        )
    )

    for row in db.execute(query).all():
        # Without a label (a detector-only project) the category names
        # the row, so a person and a vehicle are not one "unknown".
        raw = row.label or row.category
        key = (raw, row.label_taxonomy_id)
        data.names.setdefault(
            key,
            _LabelNames(
                label=raw,
                label_taxonomy_id=row.label_taxonomy_id,
                common_name=row.common_name or raw,
                scientific_name=row.scientific_name or raw,
            ),
        )
        data.observations[(row.site_id, date.fromisoformat(row.day), key)] += int(
            row.count or 0
        )
    return data


def _resolve_granularity(
    data: _ScopeData, requested: TrendGranularity | None, *, allow_day: bool
) -> TrendGranularity:
    if requested is not None:
        return requested
    if data.first is None or data.last is None:
        return "day" if allow_day else "week"
    return pick_granularity(data.first, data.last, allow_day=allow_day)


# ---------------------------------------------------------------------------
# Explore: observations per period
# ---------------------------------------------------------------------------


def get_detection_trend(
    db: Session,
    project_id: str,
    label_taxonomy_ids: list[str] | None = None,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    granularity: TrendGranularity | None = None,
) -> DetectionTrendResponse:
    """Observations per day, week or month, scoped like every dashboard card.

    A count, not a rate: see the module docstring. The periods run from
    the date filter, or else the first and last day with an observation,
    and a period without one is there as 0.
    """
    day_expr = func.strftime("%Y-%m-%d", Event.event_start_local)
    rows = db.execute(
        _scoped_observations_query(
            [day_expr.label("day"), func.sum(EventObservation.effective_count).label("count")],
            project_id, label_taxonomy_ids, site_ids, date_from, date_to,
        )
        # Date-less events (NULL bounds) have no day; exclude them.
        .where(Event.event_start_local.isnot(None))
        .group_by(day_expr)
    ).all()
    counts_by_day = {date.fromisoformat(row.day): int(row.count or 0) for row in rows}

    first = date.fromisoformat(date_from) if date_from else min(counts_by_day, default=None)
    last = date.fromisoformat(date_to) if date_to else max(counts_by_day, default=None)
    if first is None or last is None:
        return DetectionTrendResponse(
            granularity=granularity or "day", first_date=None, last_date=None, points=[]
        )
    if granularity is None:
        granularity = pick_granularity(first, last, allow_day=True)
    response = DetectionTrendResponse(
        granularity=granularity, first_date=first, last_date=last, points=[]
    )
    # A corrupt capture year: the card names the two dates instead.
    if _too_wide(first, last) or last < first:
        return response

    counts: Counter[str] = Counter()
    for day, count in counts_by_day.items():
        counts[period_of(day, granularity).key] += count
    response.points = [
        DetectionTrendPoint(period=p.key, start=p.start, end=p.end, count=counts[p.key])
        for p in build_periods(first, last, granularity)
    ]
    return response


# ---------------------------------------------------------------------------
# Trend page: the mean of site rates
# ---------------------------------------------------------------------------


def _rate_per_100(observations: int, trap_nights: int) -> float:
    return observations / trap_nights * 100


def _summary_fields(stats: RateStats) -> dict:
    return {
        "n_sites": stats.n,
        "mean": stats.mean,
        "sd": stats.sd,
        "ci_low": stats.ci_low,
        "ci_high": stats.ci_high,
    }


def _trend_periods(
    data: _ScopeData,
    span: tuple[date, date],
    granularity: TrendGranularity,
    date_from: str | None,
    date_to: str | None,
) -> list[TrendPeriod]:
    """Per period, the mean rate over the sites that ran long enough."""
    # (period key, site) -> totals within the period
    active_days: Counter[tuple[str, SiteKey]] = Counter()
    nights: Counter[tuple[str, SiteKey]] = Counter()
    observations: Counter[tuple[str, SiteKey]] = Counter()
    for site, per_day in data.nights.items():
        for day, sessions in per_day.items():
            key = (period_of(day, granularity).key, site)
            active_days[key] += 1
            nights[key] += sessions
    for (site, day, _label), count in data.observations.items():
        observations[(period_of(day, granularity).key, site)] += count

    # The share is of the whole calendar period, also where the date
    # filter cuts it: a week cut to one day would otherwise count every
    # site on a single trap night, the spike the rule is there to stop.
    # Such a piece is a gap. Its start and end still say what was in view.
    clip_start = date.fromisoformat(date_from) if date_from else None
    clip_end = date.fromisoformat(date_to) if date_to else None

    periods: list[TrendPeriod] = []
    for p in build_periods(*span, granularity):
        start = max(p.start, clip_start) if clip_start else p.start
        end = min(p.end, clip_end) if clip_end else p.end
        needed = MIN_ACTIVE_SHARE * ((p.end - p.start).days + 1)
        rates = [
            _rate_per_100(observations[(p.key, site)], nights[(p.key, site)])
            for site in data.nights
            if active_days[(p.key, site)] >= needed
        ]
        periods.append(
            TrendPeriod(
                period=p.key, start=start, end=end,
                **_summary_fields(summarise_rates(rates)),
            )
        )
    return periods


def _species_rates(data: _ScopeData) -> list[SpeciesRate]:
    """Per label, the mean rate over every site with effort in the range."""
    nights_by_site = {
        site: sum(per_day.values()) for site, per_day in data.nights.items()
    }
    # label key -> site -> observations
    by_label: dict[LabelKey, Counter[SiteKey]] = defaultdict(Counter)
    for (site, _day, key), count in data.observations.items():
        by_label[key][site] += count

    rows = [
        SpeciesRate(
            label=names.label,
            label_taxonomy_id=names.label_taxonomy_id,
            common_name=names.common_name,
            scientific_name=names.scientific_name,
            **_summary_fields(
                summarise_rates(
                    [
                        _rate_per_100(by_label[key][site], site_nights)
                        for site, site_nights in nights_by_site.items()
                    ]
                )
            ),
        )
        for key, names in data.names.items()
    ]
    rows.sort(key=lambda r: (-(r.mean or 0.0), r.label))
    return rows


def get_rate_trend(
    db: Session,
    project_id: str,
    label_taxonomy_ids: list[str] | None = None,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    granularity: TrendGranularity | None = None,
) -> RateTrendResponse:
    """The Trend page: mean site rate per period, and per species over
    the whole date range. One response, so the chart and the table are
    always the same scope."""
    data = _load_scope(db, project_id, label_taxonomy_ids, site_ids, date_from, date_to)
    granularity = _resolve_granularity(data, granularity, allow_day=False)
    span = data.span()
    return RateTrendResponse(
        granularity=granularity,
        first_date=data.first,
        last_date=data.last,
        periods=(
            _trend_periods(data, span, granularity, date_from, date_to)
            if span is not None
            else []
        ),
        species=_species_rates(data) if span is not None else [],
    )
