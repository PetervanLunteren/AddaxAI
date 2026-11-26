"""
Sequences API router.

Provides endpoints for sequence grouping, browsing, and navigation.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import file as file_crud
from app.api.crud import label_tree as label_tree_crud
from app.api.crud import sequence as sequence_crud
from app.api.crud import sequence_observation as sequence_obs_crud
from app.api.crud.sequence import VERIFY_SORT_VALUES
from app.api.schemas.sequence import (
    AdjacentSequencesResponse,
    GenerateSequencesRequest,
    GenerateSequencesResponse,
    LabelCounts,
    SequenceFilterOptions,
    SequenceObservationItem,
    SequenceSummary,
    SequenceVerificationStats,
    SequenceWithFiles,
)
from app.core.observation_attributes import Behavior, LifeStage, Sex
from app.db.base import get_db
from app.models import Project, Sequence
from app.models.sequence_observation import SequenceObservation
from app.utils.datetime_serialization import set_active_project_timezone
from app.utils.media_dates import frame_time

router = APIRouter(prefix="/api/sequences", tags=["sequences"])


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


def _set_project_tz_for_sequence(db: Session, sequence) -> None:
    """Activate project timezone from a loaded Sequence's deployment chain.

    Goes through Deployment.project directly so null-site deployments
    still resolve a timezone.
    """
    if (
        sequence.deployment
        and sequence.deployment.project
        and sequence.deployment.project.timezone
    ):
        set_active_project_timezone(sequence.deployment.project.timezone)


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


@router.post("/generate", response_model=GenerateSequencesResponse)
def generate_sequences(
    request: GenerateSequencesRequest,
    db: Session = Depends(get_db),
):
    """
    Generate or regenerate sequences for a project.

    Groups files into sequences (see `services/sequence_clustering`).
    Idempotent: deletes existing sequences before regenerating.
    """
    try:
        count = sequence_crud.generate_sequences_for_project(db, request.project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None

    return GenerateSequencesResponse(
        sequence_count=count,
        message=f"Generated {count} sequences",
    )


@router.get("/filter-options", response_model=SequenceFilterOptions)
def get_filter_options(
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_db),
):
    """Everything about the labels a project has: which, their names,
    colours and filter tree, plus the sequence date range. One pass over the
    project's boxes; the counts under a page's filters are `label-counts`."""
    if not db.query(Project.id).filter(Project.id == project_id).first():
        raise HTTPException(status_code=404, detail="Project not found")
    _set_project_tz(db, project_id)
    return sequence_crud.get_filter_options(db, project_id)


@router.get("/label-counts", response_model=LabelCounts)
def get_label_counts(
    project_id: str = Query(..., description="Project ID"),
    count_by: label_tree_crud.CountUnit = Query(..., description="Count unit"),
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
    if not db.query(Project.id).filter(Project.id == project_id).first():
        raise HTTPException(status_code=404, detail="Project not found")
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


@router.get("", response_model=list[SequenceSummary])
def list_sequences(
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
    """List sequence summaries for a project with optional filters."""
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
    return sequence_crud.get_sequences_by_project(
        db, project_id, skip=skip, limit=limit, **filters,
    )


@router.get("/count")
def get_sequence_count(
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
    """How many sequences pass the filters (`count`), and how many of those
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
    counts = sequence_crud.get_sequence_counts(db, project_id, **filters)
    return {"count": counts.total, "confirmed": counts.confirmed}


@router.get("/verification-stats", response_model=SequenceVerificationStats)
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
    """The confirmed sequences and verified cards a reprocess must keep."""
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
    return sequence_crud.get_sequence_verification_stats(
        db, project_id, **filters,
    )


@router.get("/{sequence_id}", response_model=SequenceWithFiles)
def get_sequence(
    sequence_id: str,
    db: Session = Depends(get_db),
):
    """Get sequence with all files and detections."""
    sequence = sequence_crud.get_sequence_with_files(db, sequence_id)
    if not sequence:
        raise HTTPException(status_code=404, detail="Sequence not found")

    # Set the active project timezone for datetime serialization. The
    # serializer in file.py / sequence.py uses this to emit local-with-offset
    # ISO strings for observational datetimes.
    _set_project_tz_for_sequence(db, sequence)

    # Sort files by captured_at_local (sequence order at the camera).
    # Date-less files (no EXIF capture date) sort last via the tuple key
    # so a mixed sequence never compares None against a datetime. Bursts
    # often share one second-resolution timestamp, so ties break on
    # file_path: camera filenames are sequential (IMG_0001, IMG_0002),
    # making alphabetical order the true capture order.
    sorted_files = sorted(
        sequence.files,
        key=lambda f: (
            f.captured_at_local is None,
            f.captured_at_local,
            f.file_path,
        ),
    )

    site_name = None
    if sequence.deployment and sequence.deployment.site:
        site_name = sequence.deployment.site.name

    max_n_frames = sequence_obs_crud.get_max_n_frames(db, sequence_id)
    first_seen = sequence_obs_crud.first_arrivals(
        db, sequence.deployment.project.counting_threshold, sequence_id=sequence_id
    ).get(sequence_id, {})
    observations = [
        _obs_item(obs, first_seen.get(obs.label_taxonomy_id or obs.label))
        for obs in sequence_obs_crud.list_sequence_observations(db, sequence_id)
    ]

    # The light shape per file: the cover's boxes, the cards and the
    # MaxN per species. Enough for the landing pick, the filmstrip and
    # the first still; the modal fetches the full file it focuses on.
    files = [
        file_crud.file_payload(
            db, f, "cards", camera=_camera_name(f.file_path, sequence.deployment)
        )
        for f in sorted_files
    ]

    return SequenceWithFiles(
        id=sequence.id,
        deployment_id=sequence.deployment_id,
        sequence_start_local=sequence.sequence_start_local,
        sequence_end_local=sequence.sequence_end_local,
        file_count=sequence.file_count,
        max_n_frames=max_n_frames,
        confirmed=sequence.confirmed,
        observations=observations,
        notes=sequence.notes,
        created_at_utc=sequence.created_at_utc,
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
    obs: SequenceObservation, first_arrival: datetime | None = None
) -> SequenceObservationItem:
    """Build one count-list item, resolving display names from the
    taxonomy (common_name / scientific_name) when present. `first_arrival`
    is when the species was first seen in the sequence, from
    `first_arrivals`; None when unknown."""
    tax = obs.label_taxonomy
    return SequenceObservationItem(
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


class SequenceConfirmRequest(BaseModel):
    """Set the human confirmation of a sequence's species and counts."""

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
    """Replace the sequence's human layer so its visible rows become exactly
    these cohorts."""

    observations: list[HumanObservationRow] = Field(..., min_length=1)


class SetObservationAttributesRequest(BaseModel):
    """Set sex / life stage / behaviour on one cohort row. A field not in
    the body is left alone; null clears it. Values outside the vocabularies
    in app.core.observation_attributes are refused (422)."""

    sex: Sex | None = None
    life_stage: LifeStage | None = None
    behavior: Behavior | None = None


class SequenceNotesRequest(BaseModel):
    """A person's free text about the visit. Null or blank clears it."""

    notes: str | None = Field(None, max_length=2000)


@router.patch("/{sequence_id}/confirm", response_model=SequenceWithFiles)
def confirm_sequence(
    sequence_id: str,
    body: SequenceConfirmRequest,
    db: Session = Depends(get_db),
):
    """Set/clear the human confirmation of the sequence's species and counts."""
    sequence = sequence_obs_crud.set_sequence_confirmed(db, sequence_id, body.confirmed)
    if sequence is None:
        raise HTTPException(status_code=404, detail="Sequence not found")
    return get_sequence(sequence_id, db)


@router.patch("/{sequence_id}/notes", response_model=SequenceWithFiles)
def set_sequence_notes(
    sequence_id: str,
    body: SequenceNotesRequest,
    db: Session = Depends(get_db),
):
    """Set the sequence's free text. Does not touch the confirmation."""
    sequence = sequence_obs_crud.set_sequence_notes(db, sequence_id, body.notes)
    if sequence is None:
        raise HTTPException(status_code=404, detail="Sequence not found")
    return get_sequence(sequence_id, db)


@router.post("/{sequence_id}/observations", response_model=SequenceWithFiles)
def add_sequence_observation(
    sequence_id: str,
    body: AddSpeciesRequest,
    db: Session = Depends(get_db),
):
    """Add a species the AI missed. A species the sequence already counts is
    left as it is."""
    if not db.get(Sequence, sequence_id):
        raise HTTPException(status_code=404, detail="Sequence not found")
    with _reject_unknown_taxonomy(db):
        sequence_obs_crud.add_human_species(
            db,
            sequence_id,
            category=body.category,
            count=body.count,
            label=body.label,
            label_taxonomy_id=body.label_taxonomy_id,
        )
    return get_sequence(sequence_id, db)


@router.put("/{sequence_id}/observations", response_model=SequenceWithFiles)
def set_human_observations(
    sequence_id: str,
    body: SetHumanObservationsRequest,
    db: Session = Depends(get_db),
):
    """Replace the sequence's visible cohorts with the given rows ("Same as
    last" on the Counts page)."""
    with _reject_unknown_taxonomy(db):
        sequence = sequence_obs_crud.set_human_observations(
            db, sequence_id, [row.model_dump() for row in body.observations]
        )
    if sequence is None:
        raise HTTPException(status_code=404, detail="Sequence not found")
    return get_sequence(sequence_id, db)


@router.patch(
    "/{sequence_id}/observations/{observation_id}",
    response_model=SequenceWithFiles,
)
def set_observation_count(
    sequence_id: str,
    observation_id: str,
    body: SetCountRequest,
    db: Session = Depends(get_db),
):
    """Set (or clear) the human count for one species in the sequence."""
    obs = sequence_obs_crud.set_human_count(
        db, observation_id, body.count, sequence_id=sequence_id
    )
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_sequence(sequence_id, db)


@router.patch(
    "/{sequence_id}/observations/{observation_id}/relabel",
    response_model=SequenceWithFiles,
)
def relabel_sequence_observation(
    sequence_id: str,
    observation_id: str,
    body: RelabelObservationRequest,
    db: Session = Depends(get_db),
):
    """Change the species of one count row; its count moves to the target
    (summing into the target species when it already has a row)."""
    with _reject_unknown_taxonomy(db):
        obs = sequence_obs_crud.relabel_observation(
            db,
            observation_id,
            category=body.category,
            label=body.label,
            label_taxonomy_id=body.label_taxonomy_id,
            sequence_id=sequence_id,
        )
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_sequence(sequence_id, db)


@router.patch(
    "/{sequence_id}/observations/{observation_id}/attributes",
    response_model=SequenceWithFiles,
)
def set_observation_attributes(
    sequence_id: str,
    observation_id: str,
    body: SetObservationAttributesRequest,
    db: Session = Depends(get_db),
):
    """Set sex / life stage / behaviour on one cohort row."""
    # Only the fields the body named: absent leaves alone, null clears.
    obs = sequence_obs_crud.set_observation_attributes(
        db, observation_id, sequence_id=sequence_id, **body.model_dump(exclude_unset=True)
    )
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_sequence(sequence_id, db)


@router.post(
    "/{sequence_id}/observations/{observation_id}/split",
    response_model=SequenceWithFiles,
)
def split_sequence_observation(
    sequence_id: str,
    observation_id: str,
    db: Session = Depends(get_db),
):
    """Split one row into two cohorts of the same species (source minus
    one, new row at one) so each can get its own demographics."""
    obs = sequence_obs_crud.split_observation(db, observation_id, sequence_id=sequence_id)
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_sequence(sequence_id, db)


@router.delete(
    "/{sequence_id}/observations/{observation_id}",
    response_model=SequenceWithFiles,
)
def delete_sequence_observation(
    sequence_id: str,
    observation_id: str,
    db: Session = Depends(get_db),
):
    """Remove the human contribution to one species (deletes a human-only
    row, or clears the override on an AI row)."""
    result = sequence_obs_crud.delete_sequence_observation(
        db, observation_id, sequence_id=sequence_id
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return get_sequence(sequence_id, db)


@router.post("/{sequence_id}/observations/reset", response_model=SequenceWithFiles)
def reset_sequence_counts(sequence_id: str, db: Session = Depends(get_db)):
    """Drop every human count edit on the sequence, back to the AI proposal."""
    sequence = sequence_obs_crud.reset_sequence_to_ai(db, sequence_id)
    if sequence is None:
        raise HTTPException(status_code=404, detail="Sequence not found")
    return get_sequence(sequence_id, db)


@router.get("/{sequence_id}/adjacent", response_model=AdjacentSequencesResponse)
def get_adjacent_sequences(
    sequence_id: str,
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
    """Get adjacent sequence IDs for navigation, scoped to filtered set."""
    filters = _parse_filter_params(
        site_ids, date_from, date_to, labels, verification,
        min_confidence, max_confidence,
        flagged=flagged, favorited=favorited, empty=empty,
        min_label_confidence=min_label_confidence,
        max_label_confidence=max_label_confidence,
        sort=sort, seed=seed,
    )
    _apply_project_threshold(filters, project_id, db)
    result = sequence_crud.get_adjacent_sequences(
        db, sequence_id, project_id, **filters,
    )
    return result
