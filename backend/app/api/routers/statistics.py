"""Statistics router for dashboard analytics."""

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.crud import performance as performance_crud
from app.api.crud import statistics as stats_crud
from app.api.schemas.performance import PerformanceResponse
from app.api.schemas.statistics import (
    ActivityOverlapResponse,
    ActivityPatternResponse,
    DashboardOverview,
    DetectionCategories,
    DetectionTrendPoint,
    ObservationRateMapResponse,
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
    taxonomic_rank: str | None = Query(None),
    count_mode: str = Query("events"),
    db: Session = Depends(get_db),
) -> list[SpeciesCount]:
    return stats_crud.get_species_distribution(
        db, project_id, _parse_site_ids(site_ids), date_from, date_to,
        taxonomic_rank, count_mode,
    )


@router.get("/activity-pattern", response_model=ActivityPatternResponse)
def activity_pattern(
    project_id: str = Query(...),
    species: str | None = Query(None),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    taxonomic_rank: str | None = Query(None),
    db: Session = Depends(get_db),
) -> ActivityPatternResponse:
    return stats_crud.get_activity_pattern(
        db, project_id, species, _parse_site_ids(site_ids), date_from, date_to, taxonomic_rank
    )


@router.get("/activity-overlap", response_model=ActivityOverlapResponse)
def activity_overlap(
    project_id: str = Query(..., description="Project ID"),
    species_a: str = Query(..., description="Display name of the first species"),
    species_b: str | None = Query(
        None,
        description="Display name of the second species. Omit for single-species mode.",
    ),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    taxonomic_rank: str | None = Query(
        None,
        description=(
            "Taxonomic rank for label resolution "
            "(raw|all|species|genus|family|order|class)"
        ),
    ),
    time_axis: Literal["clock", "sun"] = Query(
        "clock",
        description=(
            "clock = raw wall-clock hour; sun = Vazquez 2019 double-anchored "
            "transform using per-date sunrise / sunset from the project's "
            "timezone. Degrades to clock when no site coordinates exist."
        ),
    ),
    db: Session = Depends(get_db),
) -> ActivityOverlapResponse:
    """
    Activity overlap payload for the Plots → Activity overlap page.

    Backs a 1-or-2 species comparison with KDE curves, sun bands, diel
    classification, and (when both species have data) the Ridout &
    Linkie overlap coefficient \u0394 with a bootstrap CI.
    """
    return stats_crud.get_activity_overlap(
        db,
        project_id,
        species_a,
        species_b,
        _parse_site_ids(site_ids),
        date_from,
        date_to,
        taxonomic_rank,
        time_axis,
    )


@router.get("/detection-trend", response_model=list[DetectionTrendPoint])
def detection_trend(
    project_id: str = Query(...),
    species: str | None = Query(None),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    taxonomic_rank: str | None = Query(None),
    db: Session = Depends(get_db),
) -> list[DetectionTrendPoint]:
    return stats_crud.get_detection_trend(
        db, project_id, species, _parse_site_ids(site_ids), date_from, date_to, taxonomic_rank
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


@router.get("/observation-rate-map", response_model=ObservationRateMapResponse)
def observation_rate_map(
    project_id: str = Query(...),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    label_taxonomy_ids: str | None = Query(None),
    db: Session = Depends(get_db),
) -> ObservationRateMapResponse:
    """Per-deployment observation rate features for the Map page."""
    return stats_crud.get_observation_rate_map(
        db,
        project_id,
        _parse_site_ids(site_ids),
        date_from,
        date_to,
        _parse_site_ids(label_taxonomy_ids),
    )


def _parse_date(value: str | None, field: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as err:
        raise HTTPException(
            status_code=422, detail=f"Invalid ISO date for {field}: {value}",
        ) from err


def _parse_top_n(value: str) -> int | None:
    if value == "all":
        return None
    try:
        n = int(value)
    except ValueError as err:
        raise HTTPException(
            status_code=422, detail=f"Invalid top_n: {value}",
        ) from err
    if n <= 0:
        raise HTTPException(status_code=422, detail="top_n must be positive")
    return n


@router.get("/performance", response_model=PerformanceResponse)
def classification_performance(
    project_id: str = Query(...),
    site_ids: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    rank: Literal["class", "order", "family", "genus", "species"] = Query("species"),
    top_n: str = Query("20", description="Integer or the literal 'all'"),
    db: Session = Depends(get_db),
) -> PerformanceResponse:
    """Confusion matrix + per-class metrics for verified detections."""
    try:
        return performance_crud.get_classification_performance(
            db,
            project_id,
            site_ids=_parse_site_ids(site_ids),
            date_from=_parse_date(date_from, "date_from"),
            date_to=_parse_date(date_to, "date_to"),
            rank=rank,
            top_n=_parse_top_n(top_n),
        )
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
