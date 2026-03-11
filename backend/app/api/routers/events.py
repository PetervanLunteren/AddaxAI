"""
Events API router.

Provides endpoints for event grouping, browsing, and navigation.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.crud import event as event_crud
from app.api.crud import label_tree as label_tree_crud
from app.api.schemas.event import (
    AdjacentEventsResponse,
    EventFilterOptions,
    EventSummary,
    EventVerificationStats,
    EventWithFiles,
    GenerateEventsRequest,
    GenerateEventsResponse,
    LabelTreeResponse,
)
from app.api.schemas.file import FileWithDetections
from app.db.base import get_db

router = APIRouter(prefix="/api/events", tags=["events"])


def _parse_filter_params(
    site_ids: str | None,
    date_from: str | None,
    date_to: str | None,
    labels: str | None,
    verification: str | None,
    min_confidence: float | None,
    max_confidence: float | None,
) -> dict:
    """Parse common filter query params into kwargs for CRUD functions."""
    # Strip :unspecified suffix from label IDs (used by rolled-up taxa in tree)
    raw_labels = labels.split(",") if labels else None
    if raw_labels:
        raw_labels = [s.removesuffix(":unspecified") for s in raw_labels]

    return dict(
        site_ids=site_ids.split(",") if site_ids else None,
        date_from=datetime.fromisoformat(date_from) if date_from else None,
        date_to=datetime.fromisoformat(date_to) if date_to else None,
        labels=raw_labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )


@router.post("/generate", response_model=GenerateEventsResponse)
def generate_events(
    request: GenerateEventsRequest,
    db: Session = Depends(get_db),
):
    """
    Generate or regenerate events for a project.

    Groups files into events based on the project's independence_interval.
    Idempotent: deletes existing events before regenerating.
    """
    try:
        count = event_crud.generate_events_for_project(db, request.project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None

    return GenerateEventsResponse(
        event_count=count,
        message=f"Generated {count} events",
    )


@router.get("/filter-options", response_model=EventFilterOptions)
def get_filter_options(
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_db),
):
    """Get available filter options (distinct labels, date range) for a project."""
    return event_crud.get_filter_options(db, project_id)


@router.get("/label-tree")
def get_label_tree(
    project_id: str = Query(..., description="Project ID"),
    count_by: str = Query("event", description="Count unit: 'event' or 'detection'"),
    db: Session = Depends(get_db),
) -> LabelTreeResponse | None:
    """
    Get the label filter tree built from the label_taxonomy table.

    Returns a pre-built tree with only detected labels, annotated with counts.
    Returns null if no taxonomy data is available (frontend falls back to flat list).
    """
    result = label_tree_crud.build_label_filter_tree(project_id, db, count_by=count_by)
    if result is None:
        return None
    return result


@router.get("", response_model=list[EventSummary])
def list_events(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated labels"),
    verification: str | None = Query(None, description="Verification filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List event summaries for a project with optional filters."""
    filters = _parse_filter_params(
        site_ids, date_from, date_to, labels, verification, min_confidence, max_confidence
    )
    return event_crud.get_events_by_project(db, project_id, skip=skip, limit=limit, **filters)


@router.get("/count")
def get_event_count(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated labels"),
    verification: str | None = Query(None, description="Verification filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """Get total event count for a project with optional filters."""
    filters = _parse_filter_params(
        site_ids, date_from, date_to, labels, verification, min_confidence, max_confidence
    )
    count = event_crud.get_event_count_by_project(db, project_id, **filters)
    return {"count": count}


@router.get("/verification-stats", response_model=EventVerificationStats)
def get_verification_stats(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated labels"),
    verification: str | None = Query(None, description="Verification filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """Get aggregate file verification stats across filtered events."""
    filters = _parse_filter_params(
        site_ids,
        date_from,
        date_to,
        labels,
        verification,
        min_confidence,
        max_confidence,
    )
    return event_crud.get_event_verification_stats(db, project_id, **filters)


@router.get("/{event_id}", response_model=EventWithFiles)
def get_event(
    event_id: str,
    db: Session = Depends(get_db),
):
    """Get event with all files and detections."""
    event = event_crud.get_event_with_files(db, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Sort files by timestamp (sequence order)
    sorted_files = sorted(event.files, key=lambda f: f.timestamp)

    site_name = None
    if event.deployment and event.deployment.site:
        site_name = event.deployment.site.name

    return EventWithFiles(
        id=event.id,
        deployment_id=event.deployment_id,
        start_time=event.start_time,
        end_time=event.end_time,
        file_count=event.file_count,
        representative_file_id=event.representative_file_id,
        created_at=event.created_at,
        site_name=site_name,
        files=[FileWithDetections.model_validate(f, from_attributes=True) for f in sorted_files],
    )


@router.get("/{event_id}/adjacent", response_model=AdjacentEventsResponse)
def get_adjacent_events(
    event_id: str,
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated labels"),
    verification: str | None = Query(None, description="Verification filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """Get adjacent event IDs for navigation, scoped to filtered set."""
    filters = _parse_filter_params(
        site_ids, date_from, date_to, labels, verification, min_confidence, max_confidence
    )
    result = event_crud.get_adjacent_events(db, event_id, project_id, **filters)
    return result
