"""Statistics router for dashboard analytics."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.crud import statistics as stats_crud
from app.api.schemas.statistics import (
    ActivityPatternResponse,
    DashboardOverview,
    DetectionCategories,
    DetectionTrendPoint,
    SpeciesCount,
    VerificationProgress,
)
from app.db.base import get_db

router = APIRouter(prefix="/api/statistics", tags=["statistics"])


def _parse_site_ids(site_ids: str | None) -> list[str] | None:
    """Split comma-separated site IDs into a list, or return None."""
    if not site_ids:
        return None
    return [s.strip() for s in site_ids.split(",") if s.strip()]


@router.get("/overview", response_model=DashboardOverview)
def overview(
    project_id: str = Query(...),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: Session = Depends(get_db),
) -> DashboardOverview:
    return stats_crud.get_dashboard_overview(
        db, project_id, _parse_site_ids(site_ids), date_from, date_to
    )


@router.get("/species", response_model=list[SpeciesCount])
def species_distribution(
    project_id: str = Query(...),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: Session = Depends(get_db),
) -> list[SpeciesCount]:
    return stats_crud.get_species_distribution(
        db, project_id, _parse_site_ids(site_ids), date_from, date_to
    )


@router.get("/activity-pattern", response_model=ActivityPatternResponse)
def activity_pattern(
    project_id: str = Query(...),
    species: str | None = Query(None),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: Session = Depends(get_db),
) -> ActivityPatternResponse:
    return stats_crud.get_activity_pattern(
        db, project_id, species, _parse_site_ids(site_ids), date_from, date_to
    )


@router.get("/detection-trend", response_model=list[DetectionTrendPoint])
def detection_trend(
    project_id: str = Query(...),
    species: str | None = Query(None),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: Session = Depends(get_db),
) -> list[DetectionTrendPoint]:
    return stats_crud.get_detection_trend(
        db, project_id, species, _parse_site_ids(site_ids), date_from, date_to
    )


@router.get("/categories", response_model=DetectionCategories)
def categories(
    project_id: str = Query(...),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: Session = Depends(get_db),
) -> DetectionCategories:
    return stats_crud.get_detection_categories(
        db, project_id, _parse_site_ids(site_ids), date_from, date_to
    )


@router.get("/verification-progress", response_model=VerificationProgress)
def verification_progress(
    project_id: str = Query(...),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    db: Session = Depends(get_db),
) -> VerificationProgress:
    return stats_crud.get_verification_progress(
        db, project_id, _parse_site_ids(site_ids), date_from, date_to
    )
