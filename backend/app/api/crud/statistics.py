"""CRUD operations for dashboard statistics.

All queries are scoped to a project via the join chain:
    Detection -> File -> Deployment -> Site -> project_id
"""

from datetime import date, datetime

from sqlalchemy import Integer, Select, case, distinct, func, literal, select
from sqlalchemy.orm import Session

from app.api.schemas.statistics import (
    ActivityPatternResponse,
    DashboardOverview,
    DetectionCategories,
    DetectionTrendPoint,
    HourlyCount,
    SpeciesCount,
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
        query = query.where(File.timestamp >= date_from)
    if date_to:
        query = query.where(File.timestamp <= date_to)

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

    # For species rank, show abbreviated binomial (G. camelopardalis)
    # instead of just the epithet, matching the display_name format.
    if taxonomic_rank == "species":
        binomial = (
            func.upper(func.substr(LabelTaxonomy.taxon_genus, 1, 1))
            + literal(". ")
            + LabelTaxonomy.taxon_species
        )
        rank_display = case(
            (
                LabelTaxonomy.taxon_species.isnot(None)
                & LabelTaxonomy.taxon_genus.isnot(None),
                binomial,
            ),
            (rank_col.isnot(None), rank_col),
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


def get_trap_nights(
    db: Session,
    project_id: str,
    site_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> int:
    """Calculate total trap nights across deployments.

    For each deployment:
        effective_end = end_date or max(file timestamp date) or start_date
        nights = max(0, (effective_end - effective_start).days)

    Clips to date_from/date_to range if provided. Includes 0-file deployments.
    Returns at least 1 to avoid division by zero.
    """
    # Get deployments with their min/max file dates
    min_file_date = func.min(func.date(File.timestamp)).label("min_file_date")
    max_file_date = func.max(func.date(File.timestamp)).label("max_file_date")

    query = (
        select(
            Deployment.id,
            Deployment.start_date,
            Deployment.end_date,
            min_file_date,
            max_file_date,
        )
        .select_from(Deployment)
        .join(Site, Deployment.site_id == Site.id)
        .outerjoin(File, File.deployment_id == Deployment.id)
        .where(Site.project_id == project_id)
        .group_by(Deployment.id, Deployment.start_date, Deployment.end_date)
    )

    if site_ids:
        query = query.where(Site.id.in_(site_ids))

    rows = db.execute(query).all()

    # Parse date_from/date_to for clipping
    clip_start = date.fromisoformat(date_from) if date_from else None
    clip_end = date.fromisoformat(date_to) if date_to else None

    total_nights = 0
    for row in rows:
        dep_start = row.start_date
        if not dep_start:
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

        # Effective start: earliest of deployment start_date and first file
        effective_start = min(dep_start, min_fd) if min_fd else dep_start

        # Effective end: deployment end_date, or last file date, or start
        dep_end = row.end_date
        effective_end = dep_end or max_fd or effective_start

        # Clip to date range
        if clip_start:
            effective_start = max(effective_start, clip_start)
        if clip_end:
            effective_end = min(effective_end, clip_end)

        nights = max(0, (effective_end - effective_start).days)
        total_nights += nights

    return max(1, total_nights)


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
            func.min(File.timestamp).label("first_file_date"),
            func.max(File.timestamp).label("last_file_date"),
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
        # "Most specific": show each label at its best available Latin name.
        # Species → "G. camelopardalis", genus → "Giraffa", etc.
        # Falls back to raw label for non-animal or no-taxonomy entries.
        binomial = (
            func.upper(func.substr(LabelTaxonomy.taxon_genus, 1, 1))
            + literal(". ")
            + LabelTaxonomy.taxon_species
        )
        label_expr = case(
            (EventObservation.category != "animal", EventObservation.category),
            (
                LabelTaxonomy.taxon_species.isnot(None)
                & LabelTaxonomy.taxon_genus.isnot(None),
                binomial,
            ),
            (LabelTaxonomy.taxon_genus.isnot(None), LabelTaxonomy.taxon_genus),
            (
                LabelTaxonomy.taxon_family.isnot(None),
                LabelTaxonomy.taxon_family,
            ),
            (
                LabelTaxonomy.taxon_order.isnot(None),
                LabelTaxonomy.taxon_order,
            ),
            (
                LabelTaxonomy.taxon_class.isnot(None),
                LabelTaxonomy.taxon_class,
            ),
            else_=EventObservation.label,
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

            # For species rank, show abbreviated binomial (G. camelopardalis)
            if taxonomic_rank == "species":
                binomial = (
                    func.upper(func.substr(LabelTaxonomy.taxon_genus, 1, 1))
                    + literal(". ")
                    + LabelTaxonomy.taxon_species
                )
                rank_display = case(
                    (
                        LabelTaxonomy.taxon_species.isnot(None)
                        & LabelTaxonomy.taxon_genus.isnot(None),
                        binomial,
                    ),
                    (rank_col.isnot(None), rank_col),
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
            LabelTaxonomy.name == EventObservation.label,
        )

    if site_ids:
        query = query.where(Site.id.in_(site_ids))

    rows = db.execute(query).all()
    return [
        SpeciesCount(species=row.species, count=row.count)
        for row in rows
    ]


# ---------------------------------------------------------------------------
# 3. Activity pattern (hourly)
# ---------------------------------------------------------------------------


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
    hour_expr = func.cast(func.strftime("%H", Event.start_time), Integer)

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
            query = query.where(EventObservation.label == species)
        else:
            col_name = _RANK_COLUMNS.get(taxonomic_rank)
            if col_name:
                rank_col = getattr(LabelTaxonomy, col_name)
                has_any_taxonomy = LabelTaxonomy.taxon_class.isnot(None)
                label_expr = case(
                    (EventObservation.category != "animal", EventObservation.category),
                    (rank_col.isnot(None), rank_col),
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

    return ActivityPatternResponse(hours=hours, total_observations=total)


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
    date_expr = func.strftime("%Y-%m-%d", Event.start_time)

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
            query = query.where(EventObservation.label == species)
        else:
            col_name = _RANK_COLUMNS.get(taxonomic_rank)
            if col_name:
                rank_col = getattr(LabelTaxonomy, col_name)
                has_any_taxonomy = LabelTaxonomy.taxon_class.isnot(None)
                label_expr = case(
                    (EventObservation.category != "animal", EventObservation.category),
                    (rank_col.isnot(None), rank_col),
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
