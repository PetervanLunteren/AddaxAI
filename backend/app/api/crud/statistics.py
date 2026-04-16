"""CRUD operations for dashboard statistics.

All queries are scoped to a project via the join chain:
    Detection -> File -> Deployment -> Site -> project_id
"""

from datetime import date, datetime

from sqlalchemy import Integer, Select, and_, case, distinct, func, literal, select
from sqlalchemy.orm import Session

from app.api.schemas.statistics import (
    ActivityPatternResponse,
    DashboardOverview,
    DetectionCategories,
    DetectionTrendPoint,
    HourlyCount,
    ObservationRateMapFeature,
    ObservationRateMapResponse,
    SpeciesCount,
    SpeciesObservationCount,
    SunBands,
    VerificationProgress,
)
from app.models.deployment import Deployment
from app.models.detection import Detection
from app.models.event import Event
from app.models.event_observation import EventObservation
from app.models.file import File
from app.models.label_taxonomy import LabelTaxonomy
from app.models.project import Project
from app.models.site import Site

# ---------------------------------------------------------------------------
# Shared filter helper
# ---------------------------------------------------------------------------


def _apply_filters(
    query: Select,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> Select:
    """Apply project, site, and date filters to a joined query."""
    query = query.where(Site.project_id == project_id)

    if site_ids:
        query = query.where(Site.id.in_(site_ids))
    if date_from:
        query = query.where(File.captured_at_local >= date_from)
    if date_to:
        query = query.where(File.captured_at_local <= date_to)

    return query


def _get_detection_threshold(db: Session, project_id: str) -> float:
    """Look up the project's detection confidence threshold."""
    threshold = db.query(Project.detection_threshold).filter(
        Project.id == project_id
    ).scalar()
    return threshold if threshold is not None else 0.0


def _apply_threshold(query: Select, threshold: float) -> Select:
    """Exclude detections below threshold, but always keep verified ones."""
    from sqlalchemy import or_
    return query.where(
        or_(
            Detection.confidence >= threshold,
            Detection.verified == True,  # noqa: E712
        )
    )


# ---------------------------------------------------------------------------
# Taxonomic rank resolution
# ---------------------------------------------------------------------------

_RANK_COLUMNS = {
    "class": "taxon_class",
    "order": "taxon_order",
    "family": "taxon_family",
    "genus": "taxon_genus",
    "species": "taxon_species",
}

HIGHER_LEVEL_TAXA = "Higher-level taxa"
NO_TAXONOMY = "No taxonomy"


def _rank_display_label(taxonomic_rank: str | None):
    """Return (label_expr, needs_join) for the given taxonomic rank.

    Raw / all / None:
        label_expr = coalesce(Detection.label, Detection.category)
        needs_join  = False

    Any taxonomic rank (species/genus/family/order/class):
        label_expr = CASE expression that maps each detection to:
            - non-animals: Detection.category (person, vehicle)
            - animals with a value at the rank: that taxonomy value
            - animals with taxonomy but not at this rank: "Higher-level taxa"
            - animals with no taxonomy (bait, custom labels, etc.): "No taxonomy"
        needs_join  = True
    """
    if not taxonomic_rank or taxonomic_rank in ("raw", "all"):
        # "Most specific": show display_name (Latin) with fallback to raw label
        return (
            func.coalesce(Detection.display_name, Detection.label, Detection.category),
            False,
        )

    col_name = _RANK_COLUMNS.get(taxonomic_rank)
    if not col_name:
        return (
            func.coalesce(Detection.label, Detection.category),
            False,
        )

    rank_col = getattr(LabelTaxonomy, col_name)
    # taxon_class is the broadest rank; if it's populated, the row has
    # real taxonomy. Rows with all-null fields (level "unknown"/"none")
    # are treated as having no taxonomy.
    has_any_taxonomy = LabelTaxonomy.taxon_class.isnot(None)

    # For species rank, use the pre-computed display_name from
    # label_taxonomy (e.g. "G. camelopardalis") instead of building
    # the binomial in SQL.
    if taxonomic_rank == "species":
        rank_display = case(
            (
                LabelTaxonomy.taxon_species.isnot(None),
                LabelTaxonomy.display_name,
            ),
            else_=None,
        )
    else:
        rank_display = rank_col

    label_expr = case(
        (Detection.category != "animal", Detection.category),
        (rank_display.isnot(None), rank_display),
        (has_any_taxonomy, literal(HIGHER_LEVEL_TAXA)),
        else_=literal(NO_TAXONOMY),
    )
    return label_expr, True


# ---------------------------------------------------------------------------
# Trap nights calculation
# ---------------------------------------------------------------------------


def get_per_deployment_trap_nights(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, int]:
    """Calculate trap nights for each deployment in a project.

    For each deployment:
        effective_start = min(start_date, first file date)
        effective_end = end_date or last file date or start_date
        nights = max(0, (effective_end - effective_start).days)

    Clips to date_from/date_to range if provided. Includes 0-file deployments.
    Returns a dict mapping deployment_id to its trap nights count (raw,
    no minimum). Used by both the dashboard total and the map endpoint.
    """
    min_file_date = func.min(func.date(File.captured_at_local)).label("min_file_date")
    max_file_date = func.max(func.date(File.captured_at_local)).label("max_file_date")

    query = (
        select(
            Deployment.id,
            Deployment.start_date_local,
            Deployment.end_date_local,
            min_file_date,
            max_file_date,
        )
        .select_from(Deployment)
        .join(Site, Deployment.site_id == Site.id)
        .outerjoin(File, File.deployment_id == Deployment.id)
        .where(Site.project_id == project_id)
        .group_by(Deployment.id, Deployment.start_date_local, Deployment.end_date_local)
    )

    if site_ids:
        query = query.where(Site.id.in_(site_ids))

    rows = db.execute(query).all()

    clip_start = date.fromisoformat(date_from) if date_from else None
    clip_end = date.fromisoformat(date_to) if date_to else None

    nights_by_deployment: dict[str, int] = {}
    for row in rows:
        dep_start = row.start_date_local
        if not dep_start:
            nights_by_deployment[row.id] = 0
            continue

        # Parse file dates (SQLite returns strings)
        min_fd = row.min_file_date
        if min_fd and isinstance(min_fd, str):
            min_fd = date.fromisoformat(min_fd)
        elif min_fd and isinstance(min_fd, datetime):
            min_fd = min_fd.date()

        max_fd = row.max_file_date
        if max_fd and isinstance(max_fd, str):
            max_fd = date.fromisoformat(max_fd)
        elif max_fd and isinstance(max_fd, datetime):
            max_fd = max_fd.date()

        effective_start = min(dep_start, min_fd) if min_fd else dep_start
        dep_end = row.end_date_local
        effective_end = dep_end or max_fd or effective_start

        if clip_start:
            effective_start = max(effective_start, clip_start)
        if clip_end:
            effective_end = min(effective_end, clip_end)

        nights_by_deployment[row.id] = max(0, (effective_end - effective_start).days)

    return nights_by_deployment


def get_trap_nights(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> int:
    """Total trap nights across all deployments in a project.

    Sum of get_per_deployment_trap_nights(). Returns at least 1 so
    callers that divide by trap nights never hit a zero divisor.
    """
    per_deployment = get_per_deployment_trap_nights(
        db, project_id, site_ids, date_from, date_to
    )
    return max(1, sum(per_deployment.values()))


# ---------------------------------------------------------------------------
# 1. Dashboard overview
# ---------------------------------------------------------------------------


def get_dashboard_overview(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> DashboardOverview:
    """Aggregate counts for the top-level dashboard cards."""
    # Files and date range (not filtered by threshold)
    file_stats_query = (
        select(
            func.count(distinct(File.id)).label("total_files"),
            func.min(File.captured_at_local).label("first_file_date"),
            func.max(File.captured_at_local).label("last_file_date"),
        )
        .select_from(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
    )
    file_stats_query = _apply_filters(
        file_stats_query, project_id, site_ids, date_from, date_to,
    )
    file_stats = db.execute(file_stats_query).one()

    # Observation count (sum of MaxN across all events)
    obs_count_query = (
        select(func.coalesce(func.sum(EventObservation.max_n), 0))
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
    )
    if site_ids:
        obs_count_query = obs_count_query.where(Site.id.in_(site_ids))
    total_observations = db.execute(obs_count_query).scalar() or 0

    # Events count (Event -> Deployment -> Site)
    events_query = (
        select(func.count(distinct(Event.id)))
        .select_from(Event)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
    )
    if site_ids:
        events_query = events_query.where(Site.id.in_(site_ids))
    total_events = db.execute(events_query).scalar() or 0

    # Deployments count
    deployments_query = (
        select(func.count(distinct(Deployment.id)))
        .select_from(Deployment)
        .join(Site, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
    )
    if site_ids:
        deployments_query = deployments_query.where(Site.id.in_(site_ids))
    total_deployments = db.execute(deployments_query).scalar() or 0

    # Sites count
    sites_query = select(func.count(Site.id)).where(Site.project_id == project_id)
    if site_ids:
        sites_query = sites_query.where(Site.id.in_(site_ids))
    total_sites = db.execute(sites_query).scalar() or 0

    # Trap nights
    trap_nights = get_trap_nights(db, project_id, site_ids, date_from, date_to)

    first_date = file_stats.first_file_date
    last_date = file_stats.last_file_date

    return DashboardOverview(
        total_files=file_stats.total_files or 0,
        total_observations=total_observations,
        total_events=total_events,
        total_deployments=total_deployments,
        total_sites=total_sites,
        trap_nights=trap_nights,
        first_file_date=str(first_date) if first_date else None,
        last_file_date=str(last_date) if last_date else None,
    )


# ---------------------------------------------------------------------------
# 2. Species distribution
# ---------------------------------------------------------------------------


def get_species_distribution(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    taxonomic_rank: str | None = None,
    count_mode: str = "events",
) -> list[SpeciesCount]:
    """Top 10 labels by event count or MaxN sum.

    count_mode="events": number of independent events per label.
    count_mode="max_n": sum of MaxN across events per label.

    Taxonomic rank modes: aggregates by the requested rank using
    the label_taxonomy join on EventObservation.label.
    """
    # Build label expression for taxonomic aggregation
    if not taxonomic_rank or taxonomic_rank in ("raw", "all"):
        # "Most specific": use pre-computed display_name from label_taxonomy.
        # Falls back to raw label for non-animal or no-taxonomy entries.
        label_expr = case(
            (EventObservation.category != "animal", EventObservation.category),
            else_=func.coalesce(
                LabelTaxonomy.display_name, EventObservation.label
            ),
        )
        needs_join = True
    else:
        col_name = _RANK_COLUMNS.get(taxonomic_rank)
        if not col_name:
            label_expr = EventObservation.label
            needs_join = False
        else:
            rank_col = getattr(LabelTaxonomy, col_name)
            has_any_taxonomy = LabelTaxonomy.taxon_class.isnot(None)

            # For species rank, use pre-computed display_name
            if taxonomic_rank == "species":
                rank_display = case(
                    (
                        LabelTaxonomy.taxon_species.isnot(None),
                        LabelTaxonomy.display_name,
                    ),
                    else_=None,
                )
            else:
                rank_display = rank_col

            label_expr = case(
                (EventObservation.category != "animal", EventObservation.category),
                (rank_display.isnot(None), rank_display),
                (has_any_taxonomy, literal(HIGHER_LEVEL_TAXA)),
                else_=literal(NO_TAXONOMY),
            )
            needs_join = True

    if count_mode == "max_n":
        count_expr = func.sum(EventObservation.max_n)
    else:
        count_expr = func.count(distinct(Event.id))

    query = (
        select(
            label_expr.label("species"),
            count_expr.label("count"),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
        .group_by(label_expr)
        .order_by(count_expr.desc())
        .limit(10)
    )

    if needs_join:
        query = query.outerjoin(
            LabelTaxonomy,
            LabelTaxonomy.id == EventObservation.label_taxonomy_id,
        )

    if site_ids:
        query = query.where(Site.id.in_(site_ids))

    rows = db.execute(query).all()
    return [
        SpeciesCount(species=row.species, count=row.count)
        for row in rows
    ]


# ---------------------------------------------------------------------------
# 3. Activity pattern (hourly) + sun bands
# ---------------------------------------------------------------------------


def _avg_site_location(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
) -> tuple[float, float] | None:
    """Arithmetic mean of site coordinates in a project.

    Returns None when the project has no sites in the filtered set.
    Site.latitude / Site.longitude are NOT NULL columns, so the only
    way this returns None is if there are zero matching sites.
    """
    query = select(
        func.avg(Site.latitude).label("lat"),
        func.avg(Site.longitude).label("lon"),
    ).where(Site.project_id == project_id)
    if site_ids:
        query = query.where(Site.id.in_(site_ids))
    row = db.execute(query).one()
    if row.lat is None or row.lon is None:
        return None
    return (float(row.lat), float(row.lon))


def _reference_date_for_sun(
    date_from: str | None, date_to: str | None
) -> date:
    """Pick a single reference date for the sun-position calculation.

    Midpoint of the user's filter range when both ends are set,
    otherwise the set end, otherwise today. Matches AddaxAI-Connect.
    """
    start = date.fromisoformat(date_from) if date_from else None
    end = date.fromisoformat(date_to) if date_to else None
    if start and end:
        return start + (end - start) / 2
    return start or end or date.today()


def _compute_sun_bands(
    lat: float,
    lon: float,
    reference_date: date,
    tz_name: str,
) -> SunBands | None:
    """Compute fractional-hour dawn / sunrise / sunset / dusk at a
    location and date, in the project's local timezone.

    Uses python-astral (pure math, no network). Returns None if
    astral raises ValueError for extreme latitudes (polar night/day)
    or any other input the library refuses to process.
    """
    from zoneinfo import ZoneInfo

    from astral import LocationInfo
    from astral.sun import sun

    try:
        location = LocationInfo("project", "project", tz_name, lat, lon)
        s = sun(
            location.observer,
            date=reference_date,
            tzinfo=ZoneInfo(tz_name),
        )
    except ValueError:
        return None

    def _to_fractional_hour(dt: datetime) -> float:
        return dt.hour + dt.minute / 60 + dt.second / 3600

    return SunBands(
        dawn=_to_fractional_hour(s["dawn"]),
        sunrise=_to_fractional_hour(s["sunrise"]),
        sunset=_to_fractional_hour(s["sunset"]),
        dusk=_to_fractional_hour(s["dusk"]),
    )


def get_activity_pattern(
    db: Session,
    project_id: str,
    species: str | None = None,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    taxonomic_rank: str | None = None,
) -> ActivityPatternResponse:
    """Hourly observation counts (MaxN sum, 0-23) for activity-pattern charts."""
    hour_expr = func.cast(func.strftime("%H", Event.event_start_local), Integer)

    query = (
        select(
            hour_expr.label("hour"),
            func.sum(EventObservation.max_n).label("count"),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
        .group_by(hour_expr)
        .order_by(hour_expr)
    )

    if species:
        if not taxonomic_rank or taxonomic_rank in ("raw", "all"):
            # Filter by display name (matching species distribution output)
            display_label = case(
                (EventObservation.category != "animal", EventObservation.category),
                else_=func.coalesce(
                    LabelTaxonomy.display_name, EventObservation.label
                ),
            )
            query = query.outerjoin(
                LabelTaxonomy,
                LabelTaxonomy.name == EventObservation.label,
            )
            query = query.where(display_label == species)
        else:
            col_name = _RANK_COLUMNS.get(taxonomic_rank)
            if col_name:
                rank_col = getattr(LabelTaxonomy, col_name)
                has_any_taxonomy = LabelTaxonomy.taxon_class.isnot(None)
                # For species rank, use display_name to match
                # the abbreviated binomial shown in the dropdown
                if taxonomic_rank == "species":
                    rank_display = case(
                        (
                            LabelTaxonomy.taxon_species.isnot(None),
                            LabelTaxonomy.display_name,
                        ),
                        else_=None,
                    )
                else:
                    rank_display = rank_col
                label_expr = case(
                    (EventObservation.category != "animal", EventObservation.category),
                    (rank_display.isnot(None), rank_display),
                    (has_any_taxonomy, literal(HIGHER_LEVEL_TAXA)),
                    else_=literal(NO_TAXONOMY),
                )
                query = query.outerjoin(
                    LabelTaxonomy,
                    LabelTaxonomy.name == EventObservation.label,
                )
                query = query.where(label_expr == species)
            else:
                query = query.where(EventObservation.label == species)

    if site_ids:
        query = query.where(Site.id.in_(site_ids))

    rows = db.execute(query).all()
    counts_by_hour = {row.hour: row.count for row in rows}

    # Fill all 24 hours, inserting 0 for missing ones
    hours = [
        HourlyCount(hour=h, count=counts_by_hour.get(h, 0))
        for h in range(24)
    ]
    total = sum(hc.count for hc in hours)

    # Compute day/night bands for the chart background. Driven by the
    # project's timezone + average site lat/lon + the filter midpoint
    # date. Falls back to None if any of those are missing or astral
    # refuses to compute (polar latitudes, unknown timezone, etc.).
    sun_bands: SunBands | None = None
    tz_name = db.query(Project.timezone).filter(
        Project.id == project_id
    ).scalar()
    if tz_name:
        location = _avg_site_location(db, project_id, site_ids)
        if location is not None:
            lat, lon = location
            sun_bands = _compute_sun_bands(
                lat=lat,
                lon=lon,
                reference_date=_reference_date_for_sun(date_from, date_to),
                tz_name=tz_name,
            )

    return ActivityPatternResponse(
        hours=hours, total_observations=total, sun_bands=sun_bands
    )


# ---------------------------------------------------------------------------
# 3b. Activity overlap (Plots → Activity overlap page)
# ---------------------------------------------------------------------------


_DETECTION_TIME_CAP = 5000


def _event_decimal_hours_for_species(
    db: Session,
    project_id: str,
    species: str,
    site_ids: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    taxonomic_rank: str | None,
) -> list[tuple[float, date]]:
    """
    Pull every event observation matching one species and return
    `(decimal_hour_of_day, local_date)` tuples, each repeated max_n
    times. The repetition matches the existing get_activity_pattern
    aggregation (MaxN per event).

    Used by get_activity_overlap to build the input array for the
    circular KDE. The local date travels alongside the hour so the
    sun-time transform can look up that day's sunrise / sunset.
    """
    query = (
        select(
            Event.event_start_local.label("event_start"),
            EventObservation.max_n.label("max_n"),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
    )

    if site_ids:
        query = query.where(Site.id.in_(site_ids))
    if date_from:
        query = query.where(
            Event.event_start_local >= datetime.fromisoformat(date_from)
        )
    if date_to:
        end_of_day = datetime.combine(
            date.fromisoformat(date_to), datetime.max.time()
        )
        query = query.where(Event.event_start_local <= end_of_day)

    if not taxonomic_rank or taxonomic_rank in ("raw", "all"):
        display_label = case(
            (EventObservation.category != "animal", EventObservation.category),
            else_=func.coalesce(
                LabelTaxonomy.display_name, EventObservation.label
            ),
        )
        query = query.outerjoin(
            LabelTaxonomy, LabelTaxonomy.name == EventObservation.label
        )
        query = query.where(display_label == species)
    else:
        col_name = _RANK_COLUMNS.get(taxonomic_rank)
        if col_name:
            rank_col = getattr(LabelTaxonomy, col_name)
            has_any_taxonomy = LabelTaxonomy.taxon_class.isnot(None)
            if taxonomic_rank == "species":
                rank_display = case(
                    (
                        LabelTaxonomy.taxon_species.isnot(None),
                        LabelTaxonomy.display_name,
                    ),
                    else_=None,
                )
            else:
                rank_display = rank_col
            label_expr = case(
                (EventObservation.category != "animal", EventObservation.category),
                (rank_display.isnot(None), rank_display),
                (has_any_taxonomy, literal(HIGHER_LEVEL_TAXA)),
                else_=literal(NO_TAXONOMY),
            )
            query = query.outerjoin(
                LabelTaxonomy, LabelTaxonomy.name == EventObservation.label
            )
            query = query.where(label_expr == species)
        else:
            query = query.where(EventObservation.label == species)

    rows = db.execute(query).all()
    out: list[tuple[float, date]] = []
    for row in rows:
        dt: datetime = row.event_start
        decimal_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        local_date = dt.date()
        out.extend([(decimal_hour, local_date)] * int(row.max_n))
    return out


def _sample_size_warning(n: int) -> str | None:
    """Map n to a warning bucket for the UI badge layer."""
    if n < 30:
        return "low_n_30"
    if n < 50:
        return "low_n_50"
    if n < 75:
        return "low_n_75"
    return None


def _build_species_activity(
    label: str,
    times: list[float],
    sun_bands: SunBands | None,
    *,
    dropped_polar: int = 0,
):
    """Fit KDE, classify diel, package as a SpeciesActivity payload.

    `times` are already in the axis convention the caller wants: raw
    clock hours in clock mode, Vazquez-anchored sun hours in sun mode.
    `sun_bands` should be the bands relevant to that axis (single-ref
    clock bands in clock mode, mean-anchor bands in sun mode) so the
    diel classification matches the visible curves.
    """
    import numpy as np

    from app.api.schemas.statistics import SpeciesActivity
    from app.ml.activity_analysis import classify_diel, fit_circular_kde

    n = len(times)
    times_arr = np.asarray(times, dtype=np.float64)
    grid_hours, density = fit_circular_kde(times_arr)
    diel_class, density_by_phase = classify_diel(grid_hours, density, sun_bands)

    # Cap the rug payload so the response stays small for huge datasets.
    if n > _DETECTION_TIME_CAP:
        rng = np.random.default_rng(seed=hash(label) & 0xFFFFFFFF)
        sampled = rng.choice(times_arr, size=_DETECTION_TIME_CAP, replace=False)
        raw_for_payload = sorted(float(x) for x in sampled)
    else:
        raw_for_payload = [float(x) for x in times]

    return SpeciesActivity(
        label=label,
        n=n,
        raw_detection_times=raw_for_payload,
        kde_density=[float(x) for x in density],
        diel_class=diel_class,
        diel_density_by_phase=density_by_phase,
        sample_size_warning=_sample_size_warning(n),
        dropped_polar=dropped_polar,
    )


def get_activity_overlap(
    db: Session,
    project_id: str,
    species_a: str,
    species_b: str | None = None,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    taxonomic_rank: str | None = None,
    time_axis: str = "clock",
):
    """
    Build the full Plots → Activity overlap payload for 1 or 2 species.

    Reuses `_avg_site_location` and `_compute_sun_bands` for the sun
    overlay. Reuses `_RANK_COLUMNS` / `HIGHER_LEVEL_TAXA` / `NO_TAXONOMY`
    for taxonomic-rank-aware species filtering. The math (KDE, Δ,
    bootstrap CI, diel classification) lives in
    `app.ml.activity_analysis`; the Vazquez sun-time transform lives in
    `app.ml.sun_time`.

    `time_axis="sun"` routes each observation through the Vazquez 2019
    double-anchored transform before KDE fitting, so detections pooled
    across seasons or latitudes share a common reference frame. Degrades
    silently to clock mode when project lat / lon is missing or every
    observation's date falls in a polar window.
    """
    import numpy as np

    from app.api.schemas.statistics import (
        ActivityOverlapResponse,
        OverlapStat,
    )
    from app.ml.activity_analysis import (
        BOOTSTRAP_REPS,
        bootstrap_overlap_ci,
        estimator_label,
    )
    from app.ml.sun_time import (
        compute_anchor_bands,
        compute_anchors,
        per_date_sun_phases,
        transform_to_sun_time,
    )

    # Single-reference clock sun bands (same as get_activity_pattern).
    # Always populated when we can, independent of axis; the frontend
    # uses it for the clock-mode overlay. The tz name is echoed back
    # so the footer can show which clock the chart is in.
    sun_bands: SunBands | None = None
    tz_name = (
        db.query(Project.timezone).filter(Project.id == project_id).scalar()
    ) or "UTC"
    location = _avg_site_location(db, project_id, site_ids)
    if location is not None:
        lat, lon = location
        sun_bands = _compute_sun_bands(
            lat=lat,
            lon=lon,
            reference_date=_reference_date_for_sun(date_from, date_to),
            tz_name=tz_name,
        )

    independence_seconds = (
        db.query(Project.independence_interval)
        .filter(Project.id == project_id)
        .scalar()
        or 0
    )

    obs_a = _event_decimal_hours_for_species(
        db, project_id, species_a, site_ids, date_from, date_to, taxonomic_rank
    )
    obs_b: list[tuple[float, date]] = []
    if species_b:
        obs_b = _event_decimal_hours_for_species(
            db, project_id, species_b, site_ids, date_from, date_to, taxonomic_rank
        )

    # Decide which axis we can actually deliver. Sun mode needs a
    # project location AND at least one non-polar date across both
    # species; otherwise we silently fall back to clock.
    effective_axis: str = "clock"
    anchor_sun_bands: SunBands | None = None
    hours_a: list[float]
    hours_b: list[float]
    dropped_a = 0
    dropped_b = 0

    if time_axis == "sun" and location is not None and (obs_a or obs_b):
        lat, lon = location
        all_dates = [d for _, d in obs_a] + [d for _, d in obs_b]
        phases = per_date_sun_phases(
            all_dates, lat=lat, lon=lon, tz_name=tz_name
        )
        anchors = compute_anchors(phases)
        anchor_bands_tuple = compute_anchor_bands(phases)
        if anchors is not None and anchor_bands_tuple is not None:
            anchor_sunrise, anchor_sunset = anchors
            hours_a, dropped_a = transform_to_sun_time(
                obs_a,
                phases,
                anchor_sunrise=anchor_sunrise,
                anchor_sunset=anchor_sunset,
            )
            hours_b, dropped_b = transform_to_sun_time(
                obs_b,
                phases,
                anchor_sunrise=anchor_sunrise,
                anchor_sunset=anchor_sunset,
            )
            dawn, sunrise, sunset, dusk = anchor_bands_tuple
            anchor_sun_bands = SunBands(
                dawn=dawn, sunrise=sunrise, sunset=sunset, dusk=dusk
            )
            effective_axis = "sun"
        else:
            hours_a = [h for h, _ in obs_a]
            hours_b = [h for h, _ in obs_b]
    else:
        hours_a = [h for h, _ in obs_a]
        hours_b = [h for h, _ in obs_b]

    # Diel classification uses whichever bands match the axis: anchor
    # bands in sun mode, single-reference clock bands in clock mode.
    # This keeps the legend label consistent with the visible curves.
    diel_bands = anchor_sun_bands if effective_axis == "sun" else sun_bands

    activity_a = _build_species_activity(
        species_a, hours_a, diel_bands, dropped_polar=dropped_a
    )

    activity_b = None
    overlap = None
    if species_b:
        activity_b = _build_species_activity(
            species_b, hours_b, diel_bands, dropped_polar=dropped_b
        )

        if len(hours_a) > 0 and len(hours_b) > 0:
            delta, ci_low, ci_high = bootstrap_overlap_ci(
                np.asarray(hours_a, dtype=np.float64),
                np.asarray(hours_b, dtype=np.float64),
            )
            min_n = min(len(hours_a), len(hours_b))
            overlap = OverlapStat(
                delta_estimator=estimator_label(min_n),
                delta=delta,
                ci_low=ci_low,
                ci_high=ci_high,
                bootstrap_reps=BOOTSTRAP_REPS,
                min_n=min_n,
            )

    return ActivityOverlapResponse(
        species_a=activity_a,
        species_b=activity_b,
        overlap=overlap,
        sun_bands=sun_bands,
        anchor_sun_bands=anchor_sun_bands,
        time_axis=effective_axis,
        project_timezone=tz_name,
        independence_interval_seconds=int(independence_seconds),
    )


# ---------------------------------------------------------------------------
# 4. Detection trend (daily)
# ---------------------------------------------------------------------------


def get_detection_trend(
    db: Session,
    project_id: str,
    species: str | None = None,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    taxonomic_rank: str | None = None,
) -> list[DetectionTrendPoint]:
    """Daily observation counts (MaxN sum) for trend charts."""
    date_expr = func.strftime("%Y-%m-%d", Event.event_start_local)

    query = (
        select(
            date_expr.label("date"),
            func.sum(EventObservation.max_n).label("count"),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
        .group_by(date_expr)
        .order_by(date_expr.asc())
    )

    if species:
        if not taxonomic_rank or taxonomic_rank in ("raw", "all"):
            # Filter by display name (matching species distribution output)
            display_label = case(
                (EventObservation.category != "animal", EventObservation.category),
                else_=func.coalesce(
                    LabelTaxonomy.display_name, EventObservation.label
                ),
            )
            query = query.outerjoin(
                LabelTaxonomy,
                LabelTaxonomy.name == EventObservation.label,
            )
            query = query.where(display_label == species)
        else:
            col_name = _RANK_COLUMNS.get(taxonomic_rank)
            if col_name:
                rank_col = getattr(LabelTaxonomy, col_name)
                has_any_taxonomy = LabelTaxonomy.taxon_class.isnot(None)
                # For species rank, use display_name to match
                # the abbreviated binomial shown in the dropdown
                if taxonomic_rank == "species":
                    rank_display = case(
                        (
                            LabelTaxonomy.taxon_species.isnot(None),
                            LabelTaxonomy.display_name,
                        ),
                        else_=None,
                    )
                else:
                    rank_display = rank_col
                label_expr = case(
                    (EventObservation.category != "animal", EventObservation.category),
                    (rank_display.isnot(None), rank_display),
                    (has_any_taxonomy, literal(HIGHER_LEVEL_TAXA)),
                    else_=literal(NO_TAXONOMY),
                )
                query = query.outerjoin(
                    LabelTaxonomy,
                    LabelTaxonomy.name == EventObservation.label,
                )
                query = query.where(label_expr == species)
            else:
                query = query.where(EventObservation.label == species)

    if site_ids:
        query = query.where(Site.id.in_(site_ids))

    rows = db.execute(query).all()
    return [
        DetectionTrendPoint(date=row.date, count=row.count)
        for row in rows
    ]


# ---------------------------------------------------------------------------
# 5. Detection categories
# ---------------------------------------------------------------------------


def get_detection_categories(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> DetectionCategories:
    """Count observations (MaxN sum) by category plus blank-file count."""
    # Category counts from EventObservation (MaxN-based)
    category_query = (
        select(
            func.coalesce(
                func.sum(case(
                    (EventObservation.category == "animal", EventObservation.max_n),
                    else_=0,
                )), 0
            ).label("animal_count"),
            func.coalesce(
                func.sum(case(
                    (EventObservation.category == "person", EventObservation.max_n),
                    else_=0,
                )), 0
            ).label("person_count"),
            func.coalesce(
                func.sum(case(
                    (EventObservation.category == "vehicle", EventObservation.max_n),
                    else_=0,
                )), 0
            ).label("vehicle_count"),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(Site.project_id == project_id)
    )
    if site_ids:
        category_query = category_query.where(Site.id.in_(site_ids))
    cat_row = db.execute(category_query).one()

    # Empty (blank) file count
    empty_query = (
        select(func.count(File.id))
        .select_from(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(File.observation_type == "blank")
    )
    empty_query = _apply_filters(empty_query, project_id, site_ids, date_from, date_to)
    empty_count = db.execute(empty_query).scalar() or 0

    return DetectionCategories(
        animal_count=cat_row.animal_count or 0,
        person_count=cat_row.person_count or 0,
        vehicle_count=cat_row.vehicle_count or 0,
        empty_count=empty_count,
    )


# ---------------------------------------------------------------------------
# 6. Verification progress
# ---------------------------------------------------------------------------


def get_verification_progress(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> VerificationProgress:
    """Total vs verified file counts."""
    query = (
        select(
            func.count(File.id).label("total_files"),
            func.sum(case((File.verified == True, 1), else_=0)).label("verified_files"),  # noqa: E712
        )
        .select_from(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
    )
    query = _apply_filters(query, project_id, site_ids, date_from, date_to)
    row = db.execute(query).one()

    return VerificationProgress(
        total_files=row.total_files or 0,
        verified_files=row.verified_files or 0,
    )


# ---------------------------------------------------------------------------
# 7. Observation rate map (per-deployment GeoJSON-style features)
# ---------------------------------------------------------------------------


def _parse_iso_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def get_observation_rate_map(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    label_taxonomy_ids: list[str] | None = None,
) -> ObservationRateMapResponse:
    """Per-deployment observation rate features for the map page.

    Uses the same MaxN-per-event metric as the dashboard so rates are
    consistent across pages: rate = sum(EventObservation.max_n) /
    max(1, trap_nights) * 100.

    Each feature represents one deployment with its site coordinates,
    effort window, observation count, computed rate, and a per-species
    breakdown for the popup. Deployments that fall entirely outside
    the active filters (no events, no effort) are skipped. Deployments
    with effort but zero observations are kept so the user can see
    where they monitored without finding anything.
    """
    clip_start = _parse_iso_date(date_from)
    clip_end = _parse_iso_date(date_to)

    # 1) Per-deployment trap nights (clipped to the active date range).
    trap_nights_by_dep = get_per_deployment_trap_nights(
        db, project_id, site_ids, date_from, date_to
    )

    if not trap_nights_by_dep:
        return ObservationRateMapResponse(features=[])

    eligible_dep_ids = list(trap_nights_by_dep.keys())

    # 2) Per-deployment observation count + site/deployment metadata.
    #
    # Date and label filters go into the outer-join ON clauses, NOT a
    # WHERE. Putting them in WHERE would drop deployments that have
    # events but none matching the filter, because all their joined
    # rows would be excluded before the GROUP BY. With the filters in
    # the ON clause, non-matching events/observations simply produce
    # a NULL row under the outer join, the deployment stays in the
    # result, and sum(max_n) naturally becomes 0 for it. Effort
    # without matching observations is exactly the "empty hex" case
    # we want to render.
    event_on: list = [Event.deployment_id == Deployment.id]
    if clip_start:
        event_on.append(Event.event_start_local >= clip_start)
    if clip_end:
        end_of_day = datetime.combine(clip_end, datetime.max.time())
        event_on.append(Event.event_start_local <= end_of_day)

    obs_on: list = [EventObservation.event_id == Event.id]
    if label_taxonomy_ids:
        obs_on.append(EventObservation.label_taxonomy_id.in_(label_taxonomy_ids))

    count_query = (
        select(
            Deployment.id.label("deployment_id"),
            Deployment.site_id.label("site_id"),
            Deployment.start_date_local.label("start_date_local"),
            Deployment.end_date_local.label("end_date_local"),
            Site.name.label("site_name"),
            Site.latitude.label("latitude"),
            Site.longitude.label("longitude"),
            func.coalesce(func.sum(EventObservation.max_n), 0).label("obs_count"),
        )
        .select_from(Deployment)
        .join(Site, Deployment.site_id == Site.id)
        .outerjoin(Event, and_(*event_on))
        .outerjoin(EventObservation, and_(*obs_on))
        .where(Deployment.id.in_(eligible_dep_ids))
        .group_by(
            Deployment.id,
            Deployment.site_id,
            Deployment.start_date_local,
            Deployment.end_date_local,
            Site.name,
            Site.latitude,
            Site.longitude,
        )
    )

    rows = db.execute(count_query).all()

    # 3) Per-(deployment, label) breakdown for popups. Only fetch this if
    #    there's any data to break down.
    breakdown_by_dep: dict[str, list[SpeciesObservationCount]] = {}
    if any(row.obs_count > 0 for row in rows):
        breakdown_query = (
            select(
                Event.deployment_id.label("deployment_id"),
                EventObservation.label.label("label"),
                EventObservation.label_taxonomy_id.label("label_taxonomy_id"),
                LabelTaxonomy.display_name.label("display_name"),
                func.sum(EventObservation.max_n).label("count"),
            )
            .select_from(EventObservation)
            .join(Event, Event.id == EventObservation.event_id)
            .outerjoin(
                LabelTaxonomy,
                LabelTaxonomy.id == EventObservation.label_taxonomy_id,
            )
            .where(Event.deployment_id.in_(eligible_dep_ids))
            .group_by(
                Event.deployment_id,
                EventObservation.label,
                EventObservation.label_taxonomy_id,
                LabelTaxonomy.display_name,
            )
            .order_by(func.sum(EventObservation.max_n).desc())
        )

        if clip_start:
            breakdown_query = breakdown_query.where(Event.event_start_local >= clip_start)
        if clip_end:
            end_of_day = datetime.combine(clip_end, datetime.max.time())
            breakdown_query = breakdown_query.where(Event.event_start_local <= end_of_day)
        if label_taxonomy_ids:
            breakdown_query = breakdown_query.where(
                EventObservation.label_taxonomy_id.in_(label_taxonomy_ids)
            )

        for row in db.execute(breakdown_query).all():
            label_display = row.display_name or row.label or "unknown"
            breakdown_by_dep.setdefault(row.deployment_id, []).append(
                SpeciesObservationCount(
                    label=label_display,
                    label_taxonomy_id=row.label_taxonomy_id,
                    count=int(row.count or 0),
                )
            )

    # 4) Build the feature list. Drop deployments with neither effort nor
    #    observations (truly empty under the active filters).
    features: list[ObservationRateMapFeature] = []
    for row in rows:
        nights = trap_nights_by_dep.get(row.deployment_id, 0)
        obs = int(row.obs_count or 0)
        if nights == 0 and obs == 0:
            continue

        rate_per_100 = (obs / nights * 100) if nights > 0 else 0.0
        breakdown = breakdown_by_dep.get(row.deployment_id, [])[:10]

        features.append(
            ObservationRateMapFeature(
                deployment_id=row.deployment_id,
                site_id=row.site_id,
                site_name=row.site_name,
                latitude=row.latitude,
                longitude=row.longitude,
                start_date_local=row.start_date_local,
                end_date_local=row.end_date_local,
                trap_nights=nights,
                observation_count=obs,
                rate_per_100=rate_per_100,
                species_breakdown=breakdown,
            )
        )

    return ObservationRateMapResponse(features=features)
