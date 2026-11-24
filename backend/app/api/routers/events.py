"""
Events API router.

Provides endpoints for event grouping, browsing, and navigation.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import event as event_crud
from app.api.crud import event_observation as event_obs_crud
from app.api.crud import file as file_crud
from app.api.crud import label_tree as label_tree_crud
from app.api.crud.event import VERIFY_SORT_VALUES
from app.api.schemas.event import (
    AdjacentEventsResponse,
    EventFilterOptions,
    EventObservationItem,
    EventSummary,
    EventVerificationStats,
    EventWithFiles,
    GenerateEventsRequest,
    GenerateEventsResponse,
    LabelCounts,
)
from app.core.observation_attributes import Behavior, LifeStage, Sex
from app.db.base import get_db
from app.models import Event, Project
from app.models.event_observation import EventObservation
from app.utils.datetime_serialization import set_active_project_timezone
from app.utils.media_dates import frame_time

router = APIRouter(prefix="/api/events", tags=["events"])


def _set_project_tz(db: Session, project_id: str) -> None:
    """
    Activate the project's timezone in the request context so observational
    datetime fields get serialized with the correct local offset. See
    DEVELOPERS.md "Datetime conventions".
    """
    tz = (
        db.query(Project.timezone)
        .filter(Project.id == project_id)
        .scalar()
    )
    if tz:
        set_active_project_timezone(tz)


def _set_project_tz_for_event(db: Session, event) -> None:
    """Activate project timezone from a loaded Event's deployment chain.

    Goes through Deployment.project directly so null-site deployments
    still resolve a timezone.
    """
    if (
        event.deployment
        and event.deployment.project
        and event.deployment.project.timezone
    ):
        set_active_project_timezone(event.deployment.project.timezone)


def _parse_filter_params(
    site_ids: str | None,
    date_from: str | None,
    date_to: str | None,
    labels: str | None,
    verification: str | None,
    min_confidence: float | None,
    max_confidence: float | None,
    flagged: str | None = None,
    favorited: str | None = None,
    empty: str | None = None,
    min_label_confidence: float | None = None,
    max_label_confidence: float | None = None,
    sort: str = "newest",
    seed: int | None = None,
) -> dict:
    """Parse common filter query params into kwargs for CRUD functions."""
    parsed_labels = labels.split(",") if labels else None

    if sort not in VERIFY_SORT_VALUES:
        raise HTTPException(
            status_code=400,
            detail=f"sort must be one of: {sorted(VERIFY_SORT_VALUES)}",
        )

    return dict(
        site_ids=site_ids.split(",") if site_ids else None,
        date_from=datetime.fromisoformat(date_from) if date_from else None,
        date_to=datetime.fromisoformat(date_to) if date_to else None,
        labels=parsed_labels,
        verification=verification,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        sort=sort,
        seed=seed,
    )


def _apply_project_threshold(
    filters: dict, project_id: str, db: Session,
) -> dict:
    """Inject the project's detection threshold as `project_floor`.

    `project_floor` applies the global "threshold + verified override"
    rule: a detection is visible when `confidence >= floor OR verified`.
    `min_confidence` (the user's slider) stays untouched and is applied
    LITERALLY by CRUD — a verified low-confidence detection passes the
    floor but cannot satisfy a narrower user filter.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return filters
    filters["project_floor"] = project.counting_threshold
    return filters


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
    """Everything about the labels a project has: which, their names,
    colours and filter tree, plus the event date range. One pass over the
    project's boxes; the counts under a page's filters are `label-counts`."""
    _set_project_tz(db, project_id)
    return event_crud.get_filter_options(db, project_id)


@router.get("/label-counts", response_model=LabelCounts)
def get_label_counts(
    project_id: str = Query(..., description="Project ID"),
    count_by: str = Query(
        ..., description="Count unit: 'event', 'file', or 'detection'"
    ),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    verification: str | None = Query(None, description="Verification filter"),
    flagged: str | None = Query(None, description="Flagged filter"),
    favorited: str | None = Query(None, description="Favorited filter"),
    empty: str | None = Query(None, description="Empty filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
) -> LabelCounts:
    """How many of each label the page would show with that label ticked
    alone, under the page's other filters (the same parameters the list
    endpoints take). Keyed by taxonomy id; the tree they belong on comes
    with `filter-options`."""
    if count_by not in label_tree_crud.COUNT_UNITS:
        raise HTTPException(
            status_code=400,
            detail="count_by must be one of: event, file, detection",
        )
    counts = label_tree_crud.count_labels_under_filters(
        db,
        project_id,
        count_by,
        site_ids=site_ids.split(",") if site_ids else None,
        date_from=date_from,
        date_to=date_to,
        verification=verification,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
    )
    return LabelCounts(counts=counts)


@router.get("", response_model=list[EventSummary])
def list_events(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated labels"),
    verification: str | None = Query(None, description="Verification filter"),
    flagged: str | None = Query(None, description="Flagged filter"),
    favorited: str | None = Query(None, description="Favorited filter"),
    empty: str | None = Query(None, description="Empty filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    sort: str = Query("newest"),
    seed: int | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List event summaries for a project with optional filters."""
    _set_project_tz(db, project_id)
    filters = _parse_filter_params(
        site_ids, date_from, date_to, labels, verification,
        min_confidence, max_confidence,
        flagged=flagged, favorited=favorited, empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        sort=sort, seed=seed,
    )
    _apply_project_threshold(filters, project_id, db)
    return event_crud.get_events_by_project(
        db, project_id, skip=skip, limit=limit, **filters,
    )


@router.get("/count")
def get_event_count(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated labels"),
    verification: str | None = Query(None, description="Verification filter"),
    flagged: str | None = Query(None, description="Flagged filter"),
    favorited: str | None = Query(None, description="Favorited filter"),
    empty: str | None = Query(None, description="Empty filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """How many events pass the filters (`count`), and how many of those
    are confirmed on the Counts page (`confirmed`)."""
    filters = _parse_filter_params(
        site_ids, date_from, date_to, labels, verification,
        min_confidence, max_confidence,
        flagged=flagged, favorited=favorited, empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
    )
    _apply_project_threshold(filters, project_id, db)
    # Count is sort-invariant; the count CRUD doesn't accept sort/seed.
    filters.pop("sort", None)
    filters.pop("seed", None)
    counts = event_crud.get_event_counts(db, project_id, **filters)
    return {"count": counts.total, "confirmed": counts.confirmed}


@router.get("/verification-stats", response_model=EventVerificationStats)
def get_verification_stats(
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated labels"),
    verification: str | None = Query(None, description="Verification filter"),
    flagged: str | None = Query(None, description="Flagged filter"),
    favorited: str | None = Query(None, description="Favorited filter"),
    empty: str | None = Query(None, description="Empty filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """The confirmed events and verified cards a reprocess must keep."""
    filters = _parse_filter_params(
        site_ids,
        date_from,
        date_to,
        labels,
        verification,
        min_confidence,
        max_confidence,
        flagged=flagged,
        favorited=favorited,
        empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
    )
    _apply_project_threshold(filters, project_id, db)
    # Verification stats are sort-invariant.
    filters.pop("sort", None)
    filters.pop("seed", None)
    return event_crud.get_event_verification_stats(
        db, project_id, **filters,
    )


@router.get("/{event_id}", response_model=EventWithFiles)
def get_event(
    event_id: str,
    db: Session = Depends(get_db),
):
    """Get event with all files and detections."""
    event = event_crud.get_event_with_files(db, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Set the active project timezone for datetime serialization. The
    # serializer in file.py / event.py uses this to emit local-with-offset
    # ISO strings for observational datetimes.
    _set_project_tz_for_event(db, event)

    # Sort files by captured_at_local (sequence order at the camera).
    # Date-less files (no EXIF capture date) sort last via the tuple key
    # so a mixed event never compares None against a datetime. Bursts
    # often share one second-resolution timestamp, so ties break on
    # file_path: camera filenames are sequential (IMG_0001, IMG_0002),
    # making alphabetical order the true capture order.
    sorted_files = sorted(
        event.files,
        key=lambda f: (
            f.captured_at_local is None,
            f.captured_at_local,
            f.file_path,
        ),
    )

    site_name = None
    if event.deployment and event.deployment.site:
        site_name = event.deployment.site.name

    max_n_frames = event_obs_crud.get_max_n_frames(db, event_id)
    first_seen = event_obs_crud.first_arrivals(
        db, event.deployment.project.counting_threshold, event_id=event_id
    ).get(event_id, {})
    observations = [
        _obs_item(obs, first_seen.get(obs.label_taxonomy_id or obs.label))
        for obs in event_obs_crud.list_event_observations(db, event_id)
    ]

    # The light shape per file: the cover's boxes, the cards and the
    # MaxN per species. Enough for the landing pick, the filmstrip and
    # the first still; the modal fetches the full file it focuses on.
    files = [
        file_crud.file_payload(
            db, f, "cards", camera=_camera_name(f.file_path, event.deployment)
        )
        for f in sorted_files
    ]

    return EventWithFiles(
        id=event.id,
        deployment_id=event.deployment_id,
        event_start_local=event.event_start_local,
        event_end_local=event.event_end_local,
        file_count=event.file_count,
        max_n_frames=max_n_frames,
        confirmed=event.confirmed,
        observations=observations,
        notes=event.notes,
        created_at_utc=event.created_at_utc,
        site_name=site_name,
        files=files,
    )


def _camera_name(file_path: str, deployment) -> str | None:
    """The subfolder (camera) a file of a paired deployment sits in, else None.

    Root-level files and files outside the deployment folder give None.
    """
    if deployment is None or not deployment.paired_cameras or not deployment.folder_path:
        return None
    try:
        parts = Path(file_path).relative_to(deployment.folder_path).parts
    except ValueError:
        return None
    return parts[0] if len(parts) > 1 else None


def _obs_item(
    obs: EventObservation, first_arrival: datetime | None = None
) -> EventObservationItem:
    """Build one count-list item, resolving display names from the
    taxonomy (common_name / scientific_name) when present. `first_arrival`
    is when the species was first seen in the event, from
    `first_arrivals`; None when unknown."""
    tax = obs.label_taxonomy
    return EventObservationItem(
        id=obs.id,
        category=obs.category,
        label=obs.label,
        label_taxonomy_id=obs.label_taxonomy_id,
        common_name=(tax.common_name if tax else None) or obs.label,
        scientific_name=tax.scientific_name if tax else None,
        max_n=obs.max_n,
        effective_count=obs.effective_count,
        sex=obs.sex,
        life_stage=obs.life_stage,
        behavior=obs.behavior,
        max_n_file_id=obs.max_n_file_id,
        max_n_frame_number=obs.max_n_frame_number,
        max_n_time=(
            frame_time(obs.max_n_file, obs.max_n_frame_number)
            if obs.max_n_file_id
            else None
        ),
        first_arrival_time=first_arrival,
    )


@contextmanager
def _reject_unknown_taxonomy(db: Session) -> Iterator[None]:
    """A client-supplied label_taxonomy_id that names no row fails the
    foreign key at commit, which surfaced as a bare 500. Nothing is half
    applied (the write is one transaction), so answer 422 naming the
    field. Wraps every observation write that accepts the id. A
    project whose detector declares no primary class (a catalog error)
    is a 400 naming the model."""
    try:
        yield
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=422,
            detail="label_taxonomy_id does not name an existing taxonomy row",
        ) from e
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e


class EventConfirmRequest(BaseModel):
    """Set the human confirmation of an event's species and counts."""

    confirmed: bool


class SetCountRequest(BaseModel):
    """Set, or clear with count=null, the human count on one observation."""

    count: int | None = Field(None, ge=0)


# `category` on the three requests below: only for a detector class the
# person picked ("person", "fish"). With a species the server decides it
# (`category_after_human_edit`), so a client never has to know the
# project's detector.
class AddSpeciesRequest(BaseModel):
    """Record a species the AI missed entirely."""

    category: str | None = None
    count: int = Field(..., ge=1)
    label: str | None = None
    label_taxonomy_id: str | None = None


class RelabelObservationRequest(BaseModel):
    """Change the species of a count row (its count carries to the target)."""

    category: str | None = None
    label: str | None = None
    label_taxonomy_id: str | None = None


class HumanObservationRow(BaseModel):
    """One cohort as "Same as last" copies it: a species, its count and
    the demographics of the individuals on that row."""

    category: str | None = None
    count: int = Field(..., ge=1)
    label: str | None = None
    label_taxonomy_id: str | None = None
    sex: Sex | None = None
    life_stage: LifeStage | None = None
    behavior: Behavior | None = None


class SetHumanObservationsRequest(BaseModel):
    """Replace the event's human layer so its visible rows become exactly
    these cohorts."""

    observations: list[HumanObservationRow] = Field(..., min_length=1)


class SetObservationAttributesRequest(BaseModel):
    """Set sex / life stage / behaviour on one cohort row. A field not in
    the body is left alone; null clears it. Values outside the vocabularies
    in app.core.observation_attributes are refused (422)."""

    sex: Sex | None = None
    life_stage: LifeStage | None = None
    behavior: Behavior | None = None


class EventNotesRequest(BaseModel):
    """A person's free text about the visit. Null or blank clears it."""

    notes: str | None = Field(None, max_length=2000)


@router.patch("/{event_id}/confirm", response_model=EventWithFiles)
def confirm_event(
    event_id: str,
    body: EventConfirmRequest,
    db: Session = Depends(get_db),
):
    """Set/clear the human confirmation of the event's species and counts."""
    event = event_obs_crud.set_event_confirmed(db, event_id, body.confirmed)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return get_event(event_id, db)


@router.patch("/{event_id}/notes", response_model=EventWithFiles)
def set_event_notes(
    event_id: str,
    body: EventNotesRequest,
    db: Session = Depends(get_db),
):
    """Set the event's free text. Does not touch the confirmation."""
    event = event_obs_crud.set_event_notes(db, event_id, body.notes)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return get_event(event_id, db)


@router.post("/{event_id}/observations", response_model=EventWithFiles)
def add_event_observation(
    event_id: str,
    body: AddSpeciesRequest,
    db: Session = Depends(get_db),
):
    """Add a species the AI missed. A species the event already counts is
    left as it is."""
    if not db.get(Event, event_id):
        raise HTTPException(status_code=404, detail="Event not found")
    with _reject_unknown_taxonomy(db):
        event_obs_crud.add_human_species(
            db,
            event_id,
            category=body.category,
            count=body.count,
            label=body.label,
            label_taxonomy_id=body.label_taxonomy_id,
        )
    return get_event(event_id, db)


@router.put("/{event_id}/observations", response_model=EventWithFiles)
def set_human_observations(
    event_id: str,
    body: SetHumanObservationsRequest,
    db: Session = Depends(get_db),
):
    """Replace the event's visible cohorts with the given rows ("Same as
    last" on the Counts page)."""
    with _reject_unknown_taxonomy(db):
        event = event_obs_crud.set_human_observations(
            db, event_id, [row.model_dump() for row in body.observations]
        )
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return get_event(event_id, db)


@router.patch(
    "/{event_id}/observations/{observation_id}",
    response_model=EventWithFiles,
)
def set_observation_count(
    event_id: str,
    observation_id: str,
    body: SetCountRequest,
    db: Session = Depends(get_db),
):
    """Set (or clear) the human count for one species in the event."""
    obs = event_obs_crud.set_human_count(
        db, observation_id, body.count, event_id=event_id
    )
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_event(event_id, db)


@router.patch(
    "/{event_id}/observations/{observation_id}/relabel",
    response_model=EventWithFiles,
)
def relabel_event_observation(
    event_id: str,
    observation_id: str,
    body: RelabelObservationRequest,
    db: Session = Depends(get_db),
):
    """Change the species of one count row; its count moves to the target
    (summing into the target species when it already has a row)."""
    with _reject_unknown_taxonomy(db):
        obs = event_obs_crud.relabel_observation(
            db,
            observation_id,
            category=body.category,
            label=body.label,
            label_taxonomy_id=body.label_taxonomy_id,
            event_id=event_id,
        )
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_event(event_id, db)


@router.patch(
    "/{event_id}/observations/{observation_id}/attributes",
    response_model=EventWithFiles,
)
def set_observation_attributes(
    event_id: str,
    observation_id: str,
    body: SetObservationAttributesRequest,
    db: Session = Depends(get_db),
):
    """Set sex / life stage / behaviour on one cohort row."""
    # Only the fields the body named: absent leaves alone, null clears.
    obs = event_obs_crud.set_observation_attributes(
        db, observation_id, event_id=event_id, **body.model_dump(exclude_unset=True)
    )
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_event(event_id, db)


@router.post(
    "/{event_id}/observations/{observation_id}/split",
    response_model=EventWithFiles,
)
def split_event_observation(
    event_id: str,
    observation_id: str,
    db: Session = Depends(get_db),
):
    """Split one row into two cohorts of the same species (source minus
    one, new row at one) so each can get its own demographics."""
    obs = event_obs_crud.split_observation(db, observation_id, event_id=event_id)
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_event(event_id, db)


@router.delete(
    "/{event_id}/observations/{observation_id}",
    response_model=EventWithFiles,
)
def delete_event_observation(
    event_id: str,
    observation_id: str,
    db: Session = Depends(get_db),
):
    """Remove the human contribution to one species (deletes a human-only
    row, or clears the override on an AI row)."""
    result = event_obs_crud.delete_event_observation(db, observation_id, event_id=event_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_event(event_id, db)


@router.post("/{event_id}/observations/reset", response_model=EventWithFiles)
def reset_event_counts(event_id: str, db: Session = Depends(get_db)):
    """Drop every human count edit on the event, back to the AI proposal."""
    event = event_obs_crud.reset_event_to_ai(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return get_event(event_id, db)


@router.get("/{event_id}/adjacent", response_model=AdjacentEventsResponse)
def get_adjacent_events(
    event_id: str,
    project_id: str = Query(..., description="Project ID"),
    site_ids: str | None = Query(None, description="Comma-separated site IDs"),
    date_from: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="ISO date (YYYY-MM-DD)"),
    labels: str | None = Query(None, description="Comma-separated labels"),
    verification: str | None = Query(None, description="Verification filter"),
    flagged: str | None = Query(None, description="Flagged filter"),
    favorited: str | None = Query(None, description="Favorited filter"),
    empty: str | None = Query(None, description="Empty filter"),
    min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1),
    min_label_confidence: float | None = Query(None, ge=0, le=1),
    max_label_confidence: float | None = Query(None, ge=0, le=1),
    sort: str = Query("newest"),
    seed: int | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get adjacent event IDs for navigation, scoped to filtered set."""
    filters = _parse_filter_params(
        site_ids, date_from, date_to, labels, verification,
        min_confidence, max_confidence,
        flagged=flagged, favorited=favorited, empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        sort=sort, seed=seed,
    )
    _apply_project_threshold(filters, project_id, db)
    result = event_crud.get_adjacent_events(
        db, event_id, project_id, **filters,
    )
    return result
