"""
Events API router.

Provides endpoints for event grouping, browsing, and navigation.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.crud import event as event_crud
from app.api.schemas.event import (
    AdjacentEventsResponse,
    EventSummary,
    EventWithFiles,
    GenerateEventsRequest,
    GenerateEventsResponse,
)
from app.api.schemas.file import FileWithDetections
from app.db.base import get_db

router = APIRouter(prefix="/api/events", tags=["events"])


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
        raise HTTPException(status_code=404, detail=str(e))

    return GenerateEventsResponse(
        event_count=count,
        message=f"Generated {count} events",
    )


@router.get("", response_model=list[EventSummary])
def list_events(
    project_id: str = Query(..., description="Project ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List event summaries for a project."""
    return event_crud.get_events_by_project(db, project_id, skip=skip, limit=limit)


@router.get("/count")
def get_event_count(
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_db),
):
    """Get total event count for a project."""
    count = event_crud.get_event_count_by_project(db, project_id)
    return {"count": count}


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

    return EventWithFiles(
        id=event.id,
        deployment_id=event.deployment_id,
        start_time=event.start_time,
        end_time=event.end_time,
        file_count=event.file_count,
        representative_file_id=event.representative_file_id,
        created_at=event.created_at,
        files=[
            FileWithDetections.model_validate(f, from_attributes=True)
            for f in sorted_files
        ],
    )


@router.get("/{event_id}/adjacent", response_model=AdjacentEventsResponse)
def get_adjacent_events(
    event_id: str,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_db),
):
    """Get adjacent event IDs for navigation."""
    result = event_crud.get_adjacent_events(db, event_id, project_id)
    return result
