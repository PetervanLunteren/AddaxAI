"""CRUD operations for dashboard statistics.

All queries are scoped to a project via the join chain:
    Detection -> File -> Deployment -> Site -> project_id
"""

from sqlalchemy import Integer, Select, case, distinct, func, select
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
from app.models.file import File
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

    # Files, detections, and date range in one query
    file_stats_query = (
        select(
            func.count(distinct(File.id)).label("total_files"),
            func.count(distinct(Detection.id)).label("total_detections"),
            func.min(File.timestamp).label("first_file_date"),
            func.max(File.timestamp).label("last_file_date"),
        )
        .select_from(File)
        .join(Deployment, File.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .outerjoin(Detection, Detection.file_id == File.id)
    )
    file_stats_query = _apply_filters(file_stats_query, project_id, site_ids, date_from, date_to)
    file_stats = db.execute(file_stats_query).one()

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

    first_date = file_stats.first_file_date
    last_date = file_stats.last_file_date

    return DashboardOverview(
        total_files=file_stats.total_files or 0,
        total_detections=file_stats.total_detections or 0,
        total_events=total_events,
        total_deployments=total_deployments,
        total_sites=total_sites,
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
) -> list[SpeciesCount]:
    """Top 10 animal species by detection count."""
    query = (
        select(
            Detection.label.label("species"),
            func.count(Detection.id).label("count"),
        )
        .select_from(Detection)
        .join(File, Detection.file_id == File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .where(Detection.category == "animal")
        .where(Detection.label.isnot(None))
        .group_by(Detection.label)
        .order_by(func.count(Detection.id).desc())
        .limit(10)
    )
    query = _apply_filters(query, project_id, site_ids, date_from, date_to)

    rows = db.execute(query).all()
    return [SpeciesCount(species=row.species, count=row.count) for row in rows]


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
) -> ActivityPatternResponse:
    """Hourly detection counts (0-23) for activity-pattern charts."""
    hour_expr = func.cast(func.strftime("%H", File.timestamp), Integer)

    query = (
        select(
            hour_expr.label("hour"),
            func.count(Detection.id).label("count"),
        )
        .select_from(Detection)
        .join(File, Detection.file_id == File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .group_by(hour_expr)
        .order_by(hour_expr)
    )
    if species:
        query = query.where(Detection.label == species)
    query = _apply_filters(query, project_id, site_ids, date_from, date_to)

    rows = db.execute(query).all()
    counts_by_hour = {row.hour: row.count for row in rows}

    # Fill all 24 hours, inserting 0 for missing ones
    hours = [HourlyCount(hour=h, count=counts_by_hour.get(h, 0)) for h in range(24)]
    total = sum(hc.count for hc in hours)

    return ActivityPatternResponse(hours=hours, total_detections=total)


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
) -> list[DetectionTrendPoint]:
    """Daily detection counts for trend charts."""
    date_expr = func.strftime("%Y-%m-%d", File.timestamp)

    query = (
        select(
            date_expr.label("date"),
            func.count(Detection.id).label("count"),
        )
        .select_from(Detection)
        .join(File, Detection.file_id == File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
        .group_by(date_expr)
        .order_by(date_expr.asc())
    )
    if species:
        query = query.where(Detection.label == species)
    query = _apply_filters(query, project_id, site_ids, date_from, date_to)

    rows = db.execute(query).all()
    return [DetectionTrendPoint(date=row.date, count=row.count) for row in rows]


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
    """Count detections by category plus blank-file count."""

    # Detection category counts
    category_query = (
        select(
            func.sum(case((Detection.category == "animal", 1), else_=0)).label("animal_count"),
            func.sum(case((Detection.category == "person", 1), else_=0)).label("person_count"),
            func.sum(case((Detection.category == "vehicle", 1), else_=0)).label("vehicle_count"),
        )
        .select_from(Detection)
        .join(File, Detection.file_id == File.id)
        .join(Deployment, File.deployment_id == Deployment.id)
        .join(Site, Deployment.site_id == Site.id)
    )
    category_query = _apply_filters(category_query, project_id, site_ids, date_from, date_to)
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
