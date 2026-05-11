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
    LabelProgressRow,
    ObservationRateMapFeature,
    ObservationRateMapResponse,
    SpeciesCount,
    SpeciesObservationCount,
    SunBands,
    VerificationProgress,
    VerificationProgressByLabel,
)
from app.ml.taxonomic_rank import HIGHER_LEVEL_TAXA, NO_TAXONOMY
from app.ml.taxonomic_rank import RANK_COLUMNS as _RANK_COLUMNS
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
    """Apply project, site, and date filters to a joined query.

    Expects Deployment to already be present in the join chain so the
    filter can hit Deployment.project_id and Deployment.site_id
    directly. site_ids may include the NO_SITE_SENTINEL reserved token
    to match deployments whose site_id is NULL.
    """
    from app.api.crud.deployment import site_ids_filter

    query = query.where(Deployment.project_id == project_id)

    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.where(site_clause)
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
        # The taxon_* columns are stored lowercase (CSV convention).
        # Family / genus / order / class names are conventionally
        # capitalised, so upper-case the first letter for display.
        # Mirrors app.ml.taxonomic_rank.to_display_case.
        rank_display = func.upper(func.substr(rank_col, 1, 1)).concat(
            func.substr(rank_col, 2)
        )

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
    """Folder-aware trap-nights count per deployment in a project.

    Delegates to `compute_trap_nights_for_deployments` which buckets each
    deployment's files by folder (SD-card boundary) and sums per-folder
    `(max - min) + 1` day spans. Empty deployments get 0.
    """
    from app.api.crud.deployment import site_ids_filter
    from app.api.crud.trap_nights import compute_trap_nights_for_deployments

    query = select(Deployment.id).where(Deployment.project_id == project_id)
    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.where(site_clause)
    deployment_ids = [row[0] for row in db.execute(query).all()]

    clip_start = date.fromisoformat(date_from) if date_from else None
    clip_end = date.fromisoformat(date_to) if date_to else None

    nights = compute_trap_nights_for_deployments(
        db, deployment_ids, clip_start=clip_start, clip_end=clip_end
    )
    # Ensure every deployment is represented, including those with no files.
    return {dep_id: nights.get(dep_id, 0) for dep_id in deployment_ids}


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
    from app.api.crud.deployment import site_ids_filter

    # Files and date range (not filtered by threshold)
    file_stats_query = (
        select(
            func.count(distinct(File.id)).label("total_files"),
            func.min(File.captured_at_local).label("first_file_date"),
            func.max(File.captured_at_local).label("last_file_date"),
        )
        .select_from(File)
        .join(Deployment, File.deployment_id == Deployment.id)
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
        .where(Deployment.project_id == project_id)
    )
    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        obs_count_query = obs_count_query.where(site_clause)
    total_observations = db.execute(obs_count_query).scalar() or 0

    # Events count (Event -> Deployment)
    events_query = (
        select(func.count(distinct(Event.id)))
        .select_from(Event)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    )
    if site_clause is not None:
        events_query = events_query.where(site_clause)
    total_events = db.execute(events_query).scalar() or 0

    # Deployments count
    deployments_query = (
        select(func.count(distinct(Deployment.id)))
        .select_from(Deployment)
        .where(Deployment.project_id == project_id)
    )
    if site_clause is not None:
        deployments_query = deployments_query.where(site_clause)
    total_deployments = db.execute(deployments_query).scalar() or 0

    # Sites count. Ignore NO_SITE_SENTINEL here (the sentinel has no
    # meaning for counting sites themselves).
    from app.api.crud.deployment import NO_SITE_SENTINEL

    sites_query = select(func.count(Site.id)).where(Site.project_id == project_id)
    if site_ids:
        real_site_ids = [s for s in site_ids if s != NO_SITE_SENTINEL]
        if real_site_ids:
            sites_query = sites_query.where(Site.id.in_(real_site_ids))
        else:
            sites_query = sites_query.where(Site.id.is_(None))  # always false
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

    from app.api.crud.deployment import site_ids_filter

    query = (
        select(
            label_expr.label("species"),
            count_expr.label("count"),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
        .group_by(label_expr)
        .order_by(count_expr.desc())
        .limit(10)
    )

    if needs_join:
        query = query.outerjoin(
            LabelTaxonomy,
            LabelTaxonomy.id == EventObservation.label_taxonomy_id,
        )

    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.where(site_clause)

    rows = db.execute(query).all()
    return [
        SpeciesCount(species=row.species, count=row.count)
        for row in rows
    ]


# ---------------------------------------------------------------------------
# 3. Activity pattern (hourly) + sun bands
# ---------------------------------------------------------------------------


def _count_deployments_without_site(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
) -> int:
    """
    How many deployments in the filtered project set have no site?

    Used by GPS-dependent dashboards so the UI can render a banner.
    Ignores the NO_SITE_SENTINEL if present in site_ids (a user
    explicitly filtering TO the no-site set already knows about them).
    """
    from app.api.crud.deployment import NO_SITE_SENTINEL

    query = (
        select(func.count(Deployment.id))
        .where(Deployment.project_id == project_id)
        .where(Deployment.site_id.is_(None))
    )
    if site_ids:
        real_ids = [s for s in site_ids if s != NO_SITE_SENTINEL]
        if real_ids:
            # The user picked specific sites; null-site deployments
            # are already filtered out of the result set, so none are
            # "skipped".
            return 0
    return int(db.scalar(query) or 0)


def _avg_site_location(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
) -> tuple[float, float] | None:
    """Arithmetic mean of site coordinates in a project.

    Returns None when the project has no sites in the filtered set.
    Site.latitude / Site.longitude are NOT NULL columns, so the only
    way this returns None is if there are zero matching sites. The
    NO_SITE_SENTINEL is ignored here: deployments without a site have
    no coordinates to contribute.
    """
    from app.api.crud.deployment import NO_SITE_SENTINEL

    query = select(
        func.avg(Site.latitude).label("lat"),
        func.avg(Site.longitude).label("lon"),
    ).where(Site.project_id == project_id)
    if site_ids:
        real_ids = [s for s in site_ids if s != NO_SITE_SENTINEL]
        if not real_ids:
            return None
        query = query.where(Site.id.in_(real_ids))
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
        .where(Deployment.project_id == project_id)
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

    from app.api.crud.deployment import site_ids_filter

    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.where(site_clause)

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
        hours=hours,
        total_observations=total,
        sun_bands=sun_bands,
        deployments_without_site=_count_deployments_without_site(
            db, project_id, site_ids
        ),
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
    from app.api.crud.deployment import site_ids_filter

    query = (
        select(
            Event.event_start_local.label("event_start"),
            EventObservation.max_n.label("max_n"),
        )
        .select_from(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .where(Deployment.project_id == project_id)
    )

    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.where(site_clause)
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
    # so the footer can show which clock the chart is in. The reference
    # date (filter midpoint) is also echoed so the chart can caption
    # which day's sun events the bands represent.
    sun_bands: SunBands | None = None
    sun_bands_reference_date: date | None = None
    tz_name = (
        db.query(Project.timezone).filter(Project.id == project_id).scalar()
    ) or "UTC"
    location = _avg_site_location(db, project_id, site_ids)
    if location is not None:
        lat, lon = location
        reference_date = _reference_date_for_sun(date_from, date_to)
        sun_bands = _compute_sun_bands(
            lat=lat,
            lon=lon,
            reference_date=reference_date,
            tz_name=tz_name,
        )
        if sun_bands is not None:
            sun_bands_reference_date = reference_date

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
        sun_bands_reference_date=sun_bands_reference_date,
        anchor_sun_bands=anchor_sun_bands,
        time_axis=effective_axis,
        project_timezone=tz_name,
        independence_interval_seconds=int(independence_seconds),
        deployments_without_site=_count_deployments_without_site(
            db, project_id, site_ids
        ),
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
        .where(Deployment.project_id == project_id)
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

    from app.api.crud.deployment import site_ids_filter

    site_clause = site_ids_filter(site_ids)
    if site_clause is not None:
        query = query.where(site_clause)

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
    """Count captures (still images and extracted video frames) per category.

    Each capture is attributed to a single category by its
    `observation_type` column, which encodes the priority rule
    animal > human > vehicle > blank (see
    `recalculate_observation_type` in `crud/file.py`). The four counts
    therefore partition the captures: a photo with both an animal and
    a person lands in Animals only, never both. The sum can be at most
    the total capture count, never above it.
    """
    capture_types = ("image", "frame")

    query = (
        select(
            File.observation_type.label("obs_type"),
            func.count(File.id).label("n"),
        )
        .select_from(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .where(File.file_type.in_(capture_types))
        .group_by(File.observation_type)
    )
    query = _apply_filters(query, project_id, site_ids, date_from, date_to)
    counts: dict[str, int] = {
        row.obs_type: int(row.n or 0) for row in db.execute(query).all()
    }

    return DetectionCategories(
        animal_count=counts.get("animal", 0),
        person_count=counts.get("human", 0),
        vehicle_count=counts.get("vehicle", 0),
        empty_count=counts.get("blank", 0),
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
    )
    query = _apply_filters(query, project_id, site_ids, date_from, date_to)
    row = db.execute(query).one()

    return VerificationProgress(
        total_files=row.total_files or 0,
        verified_files=row.verified_files or 0,
    )


def get_verification_progress_by_label(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> VerificationProgressByLabel:
    """Per-class verified vs total detection counts.

    One row per (label_taxonomy_id, category). Each detection has
    exactly one label so rows partition cleanly. Counts respect the
    project's detection threshold (floor with the verified override).
    `false detection` rows are excluded since they are not a real class.
    Sorted by total descending so the highest-support classes come first.
    """
    threshold = _get_detection_threshold(db, project_id)

    query = (
        select(
            Detection.label_taxonomy_id.label("label_taxonomy_id"),
            Detection.category.label("category"),
            LabelTaxonomy.display_name.label("display_name"),
            func.count(Detection.id).label("total"),
            func.sum(
                case((Detection.verified == True, 1), else_=0)  # noqa: E712
            ).label("verified"),
        )
        .select_from(Detection)
        .join(File, File.id == Detection.file_id)
        .join(Deployment, Deployment.id == File.deployment_id)
        .outerjoin(
            LabelTaxonomy,
            LabelTaxonomy.id == Detection.label_taxonomy_id,
        )
        .where(
            (Detection.label.is_(None)) | (Detection.label != "false detection"),
        )
        .group_by(
            Detection.label_taxonomy_id,
            Detection.category,
            LabelTaxonomy.display_name,
        )
        .order_by(func.count(Detection.id).desc())
    )
    query = _apply_filters(query, project_id, site_ids, date_from, date_to)
    query = _apply_threshold(query, threshold)

    rows: list[LabelProgressRow] = []
    for row in db.execute(query).all():
        display = row.display_name or (row.category or "unknown").capitalize()
        rows.append(
            LabelProgressRow(
                label_taxonomy_id=row.label_taxonomy_id,
                display_name=display,
                verified=int(row.verified or 0),
                total=int(row.total or 0),
            )
        )
    return VerificationProgressByLabel(rows=rows)


# ---------------------------------------------------------------------------
# 7. Observation rate map (per-site GeoJSON-style features)
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
    """Per-site observation rate features for the map page.

    Sites are the spatial unit (the camera stays put across multiple
    deployments / SD-card pulls). Each feature represents one site
    with summed trap nights and observation counts across all of its
    deployments that pass the active filters.

    Uses the same MaxN-per-event metric as the dashboard so rates are
    consistent across pages: rate = sum(EventObservation.max_n) /
    max(1, trap_nights) * 100. Sites with no effort and no observations
    under the active filters are skipped. Sites with effort but zero
    matching observations are kept (hollow markers) so the user can see
    where they monitored without finding anything.

    `earliest_start_local` and `latest_end_local` describe the full
    monitoring range across the site's contributing deployments and
    are NOT clipped to the filter window. Deployments without a
    `site_id` are counted into `deployments_without_site` for a banner.
    """
    clip_start = _parse_iso_date(date_from)
    clip_end = _parse_iso_date(date_to)

    # 1) Per-deployment trap nights (clipped to the active date range).
    #    Aggregated up to per-site in step 4.
    trap_nights_by_dep = get_per_deployment_trap_nights(
        db, project_id, site_ids, date_from, date_to
    )

    if not trap_nights_by_dep:
        return ObservationRateMapResponse(features=[])

    eligible_dep_ids = list(trap_nights_by_dep.keys())

    # 2) Resolve each eligible deployment back to its site and date
    #    window. One query so the feature builder can group in Python
    #    without N round-trips. Deployments with site_id IS NULL are
    #    counted into the banner total here.
    dep_meta_rows = db.execute(
        select(
            Deployment.id,
            Deployment.site_id,
            Deployment.start_date_local,
            Deployment.end_date_local,
        ).where(Deployment.id.in_(eligible_dep_ids))
    ).all()

    site_id_by_dep: dict[str, str | None] = {}
    start_by_dep: dict[str, date] = {}
    end_by_dep: dict[str, date | None] = {}
    deployments_without_site = 0
    for r in dep_meta_rows:
        site_id_by_dep[r.id] = r.site_id
        start_by_dep[r.id] = r.start_date_local
        end_by_dep[r.id] = r.end_date_local
        if r.site_id is None:
            deployments_without_site += 1

    # 3) Per-site observation count + site metadata.
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

    site_rows_query = (
        select(
            Site.id.label("site_id"),
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
        .group_by(Site.id, Site.name, Site.latitude, Site.longitude)
    )

    site_rows = db.execute(site_rows_query).all()

    # 4) Per-(site, label) breakdown for popups. Only fetch this if
    #    there's any data to break down.
    breakdown_by_site: dict[str, list[SpeciesObservationCount]] = {}
    if any(row.obs_count > 0 for row in site_rows):
        breakdown_query = (
            select(
                Deployment.site_id.label("site_id"),
                EventObservation.label.label("label"),
                EventObservation.label_taxonomy_id.label("label_taxonomy_id"),
                LabelTaxonomy.display_name.label("display_name"),
                func.sum(EventObservation.max_n).label("count"),
            )
            .select_from(EventObservation)
            .join(Event, Event.id == EventObservation.event_id)
            .join(Deployment, Deployment.id == Event.deployment_id)
            .outerjoin(
                LabelTaxonomy,
                LabelTaxonomy.id == EventObservation.label_taxonomy_id,
            )
            .where(Deployment.id.in_(eligible_dep_ids))
            .where(Deployment.site_id.isnot(None))
            .group_by(
                Deployment.site_id,
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
            breakdown_by_site.setdefault(row.site_id, []).append(
                SpeciesObservationCount(
                    label=label_display,
                    label_taxonomy_id=row.label_taxonomy_id,
                    count=int(row.count or 0),
                )
            )

    # 5) Pre-compute deployments grouped by site for nights / date roll-up.
    deps_by_site: dict[str, list[str]] = {}
    for dep_id, site_id in site_id_by_dep.items():
        if site_id is None:
            continue
        deps_by_site.setdefault(site_id, []).append(dep_id)

    # 6) Build the feature list. Drop sites with neither effort nor
    #    observations (truly empty under the active filters).
    features: list[ObservationRateMapFeature] = []
    for row in site_rows:
        member_dep_ids = deps_by_site.get(row.site_id, [])
        nights = sum(trap_nights_by_dep.get(d, 0) for d in member_dep_ids)
        obs = int(row.obs_count or 0)
        if nights == 0 and obs == 0:
            continue

        starts = [start_by_dep[d] for d in member_dep_ids if start_by_dep.get(d)]
        ends = [end_by_dep[d] for d in member_dep_ids if end_by_dep.get(d) is not None]
        if not starts:
            # Defensive: should never happen; deployments always have
            # start_date_local once analysed. Skip rather than crash.
            continue

        rate_per_100 = (obs / nights * 100) if nights > 0 else 0.0
        breakdown = breakdown_by_site.get(row.site_id, [])[:10]

        features.append(
            ObservationRateMapFeature(
                site_id=row.site_id,
                site_name=row.site_name,
                latitude=row.latitude,
                longitude=row.longitude,
                deployment_count=len(member_dep_ids),
                earliest_start_local=min(starts),
                latest_end_local=max(ends) if ends else None,
                trap_nights=nights,
                observation_count=obs,
                rate_per_100=rate_per_100,
                species_breakdown=breakdown,
            )
        )

    features.sort(key=lambda f: f.site_name.lower())

    return ObservationRateMapResponse(
        features=features,
        deployments_without_site=deployments_without_site,
    )
