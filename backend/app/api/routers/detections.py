"""
Detections API router.

Provides endpoints for creating, updating, and deleting detections
(human-drawn annotations), crop thumbnails, and detection-level verification.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.crud import detection as detection_crud
from app.api.crud import file as file_crud
from app.api.crud.event_observation import (
    get_event_ids_for_detections,
    get_project_threshold_for_detections,
    recalculate_max_n_for_events,
)
from app.api.schemas.detection import (
    DetectionCreateHuman,
    DetectionResponse,
    DetectionUpdate,
)
from app.db.base import get_db
from app.models import Detection, File
from app.services.crop_service import get_or_create_crop, invalidate_crop_cache

router = APIRouter(prefix="/api/detections", tags=["detections"])


def _recalculate_max_n(db: Session, detection_ids: list[str]) -> None:
    """Recalculate MaxN for events affected by the given detections, and commit.

    **The commit is unconditional, and that is load-bearing.** `get_db` only
    closes the session, it never commits, so whatever is still pending when
    the endpoint returns is thrown away. This helper is the last call in every
    route that uses it, and `recompute_file_verified` runs before it without
    committing on purpose ("the caller owns the transaction").

    While the commit sat behind an early `return` for "no events", drawing a
    box on a file that belongs to no event saved the detection and dropped the
    rollup beside it: `File.verified` stayed FALSE for a photo the person had
    just judged, and `addaxai-files.csv` exported it that way. A file has no
    event only when event generation never reached it, e.g. an analysis run
    that failed part way.

    Pinned by `test_drawing_a_box_marks_the_file_verified_past_the_request`.
    """
    event_ids = get_event_ids_for_detections(db, detection_ids)
    if event_ids:
        threshold = get_project_threshold_for_detections(db, detection_ids)
        recalculate_max_n_for_events(db, event_ids, threshold)
    db.commit()


@router.post("", response_model=DetectionResponse, status_code=201)
def create_detection(
    data: DetectionCreateHuman,
    db: Session = Depends(get_db),
):
    """
    Create a human-drawn detection.

    Sets classification_method="human", confidence=1.0, job_id=None.

    A file id that no longer exists is refused here rather than left to the
    foreign key, which surfaces as a 500 the user cannot act on. The grid
    holds ids fetched earlier, so a deployment deleted in the meantime is
    the ordinary way to arrive with a stale one.
    """
    if db.get(File, data.file_id) is None:
        raise HTTPException(status_code=404, detail="File not found")

    detection = detection_crud.create_human_detection(db, data)
    file_crud.recalculate_observation_type(db, data.file_id)
    file_crud.recompute_file_verified(db, [data.file_id])
    _recalculate_max_n(db, [detection.id])
    return detection


@router.patch("/{detection_id}", response_model=DetectionResponse)
def update_detection(
    detection_id: str,
    update: DetectionUpdate,
    db: Session = Depends(get_db),
):
    """
    Update a detection's category, bounding box, or label.

    Sets classification_method="human" when label is edited.
    Invalidates crop cache when bbox changes.
    """
    detection = detection_crud.update_detection(db, detection_id, update)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    # Recalculate observation type when the category *or* the label changed.
    # Both feed it: a "false detection" label takes the box out of the running
    # for what the file is about, so a label-only edit that leaves the category
    # alone still moves the file to "blank". Bulk relabel already covers this;
    # this path used to skip it and leave observation_type stale.
    if update.category is not None or "label" in update.model_fields_set:
        file_crud.recalculate_observation_type(db, detection.file_id)

    # Invalidate crop cache if bbox changed
    if any(
        getattr(update, f) is not None
        for f in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")
    ):
        invalidate_crop_cache(detection_id)

    _recalculate_max_n(db, [detection_id])
    return detection


@router.delete("/by-file/{file_id}")
def delete_detections_by_file(
    file_id: str,
    db: Session = Depends(get_db),
):
    """Delete all detections for a file."""
    # Capture affected events and threshold before deletion
    det_ids = [
        d.id for d in db.query(Detection.id).filter(
            Detection.file_id == file_id
        ).all()
    ]
    affected_event_ids = get_event_ids_for_detections(db, det_ids) if det_ids else []
    # Only looked up when it will actually be used, and always before the
    # delete: the lookup joins through the detection rows, so afterwards
    # there is nothing to resolve. This used to be `... else 0.0`, a value
    # that reads as "no threshold" but means "every detection passes".
    threshold = (
        get_project_threshold_for_detections(db, det_ids)
        if affected_event_ids
        else None
    )
    count = detection_crud.delete_detections_by_file(db, file_id)
    file_crud.recompute_file_verified(db, [file_id])
    file_crud.recalculate_observation_type(db, file_id)
    if affected_event_ids:
        recalculate_max_n_for_events(db, affected_event_ids, threshold)
    # Outside the branch, for the reason spelled out in `_recalculate_max_n`:
    # the rollup above does not commit, and a request that returns with it
    # pending loses it.
    db.commit()
    return {"deleted_count": count}


@router.delete("/{detection_id}", status_code=204)
def delete_detection(
    detection_id: str,
    db: Session = Depends(get_db),
):
    """Delete a detection."""
    # Get detection first to know file_id for recalculation
    detection = detection_crud.get_detection(db, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    file_id = detection.file_id
    # Capture event IDs and threshold before deletion
    affected_event_ids = get_event_ids_for_detections(db, [detection_id])
    threshold = get_project_threshold_for_detections(db, [detection_id])
    detection_crud.delete_detection(db, detection_id)
    file_crud.recompute_file_verified(db, [file_id])
    file_crud.recalculate_observation_type(db, file_id)
    if affected_event_ids:
        recalculate_max_n_for_events(db, affected_event_ids, threshold)
    db.commit()


# --- Crop endpoint ---


@router.get("/{detection_id}/crop")
def get_detection_crop(
    detection_id: str,
    size: int = Query(200, ge=32, le=512, description="Crop size in pixels"),
    db: Session = Depends(get_db),
):
    """
    Serve a cropped thumbnail of a detection.

    Generates and caches a square JPEG crop from the source image
    at the detection's bounding box location.
    """
    jpeg_bytes = get_or_create_crop(detection_id, size, db)
    if not jpeg_bytes:
        raise HTTPException(status_code=404, detail="Could not generate crop")

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# --- Detection verification endpoints ---


class VerifyRequest(BaseModel):
    verified: bool = True


class BulkVerifyRequest(BaseModel):
    detection_ids: list[str] = Field(..., max_length=500)
    verified: bool = True


class BulkRelabelRequest(BaseModel):
    detection_ids: list[str] = Field(..., max_length=500)
    label: str | None = None
    category: str | None = None


class BulkRevertRequest(BaseModel):
    detection_ids: list[str] = Field(..., max_length=500)


class BulkDismissRequest(BaseModel):
    detection_ids: list[str] = Field(..., max_length=500)
    # True hides the cohort from suggestions; False undoes a dismiss.
    dismissed: bool = True


@router.patch("/{detection_id}/verify", response_model=DetectionResponse)
def verify_detection(
    detection_id: str,
    body: VerifyRequest,
    db: Session = Depends(get_db),
):
    """Verify or unverify a single detection."""
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    detection.verified = body.verified
    detection.verified_at_utc = datetime.now(UTC) if body.verified else None
    file_crud.recompute_file_verified(db, [detection.file_id])
    db.commit()
    # Verifying makes a detection "pass" regardless of confidence, which can
    # flip the file's observation_type (it counts over-threshold OR verified).
    file_crud.recalculate_observation_type(db, detection.file_id)
    db.refresh(detection)
    _recalculate_max_n(db, [detection_id])
    return detection


@router.post("/bulk-verify")
def bulk_verify_detections(
    body: BulkVerifyRequest,
    db: Session = Depends(get_db),
):
    """Bulk verify/unverify detections (max 500)."""
    now = datetime.now(UTC) if body.verified else None
    updated = (
        db.query(Detection)
        .filter(Detection.id.in_(body.detection_ids))
        .update(
            {"verified": body.verified, "verified_at_utc": now},
            synchronize_session="fetch",
        )
    )
    file_crud.recompute_file_verified_for_detections(db, body.detection_ids)
    db.commit()
    # Verifying can flip observation_type (verified detections always pass),
    # so re-derive it for every touched file.
    file_ids = {
        fid
        for (fid,) in db.query(Detection.file_id)
        .filter(Detection.id.in_(body.detection_ids))
        .all()
    }
    for fid in file_ids:
        file_crud.recalculate_observation_type(db, fid)
    _recalculate_max_n(db, body.detection_ids)
    return {"updated_count": updated}


@router.post("/bulk-dismiss")
def bulk_dismiss_detections(
    body: BulkDismissRequest,
    db: Session = Depends(get_db),
):
    """Dismiss/undismiss a cohort of suggestions (max 500).

    Sets `suggestion_dismissed`, which hides the detections from the
    suggestions review (toolbar pill, cohort dividers, suggestions-sort
    grid). It touches neither `label` nor `verified`, so there is no
    max_n or file-verified recompute to do.
    """
    updated = (
        db.query(Detection)
        .filter(Detection.id.in_(body.detection_ids))
        .update(
            {"suggestion_dismissed": body.dismissed},
            synchronize_session="fetch",
        )
    )
    db.commit()
    return {"updated_count": updated}


@router.post("/bulk-relabel")
def bulk_relabel_detections(
    body: BulkRelabelRequest,
    db: Session = Depends(get_db),
):
    """Bulk relabel detections (max 500). Sets classification_method='human'."""
    # Nothing asked for is nothing done, not a missing row. Answered before
    # the query so the code below can assume a non-empty list (it reads
    # `detections[0]` to resolve the taxonomy). Matches bulk-verify.
    if not body.detection_ids:
        return {"updated_count": 0}

    detections = (
        db.query(Detection)
        .filter(Detection.id.in_(body.detection_ids))
        .all()
    )
    if not detections:
        raise HTTPException(status_code=404, detail="No detections found")

    # Resolve taxonomy ID for the new label
    new_taxonomy_id = None
    if body.label:
        from app.api.crud.detection import _resolve_detection_taxonomy
        new_taxonomy_id = _resolve_detection_taxonomy(db, detections[0], body.label)

    # Resolve both names from the taxonomy row (single source of truth)
    new_scientific_name = None
    new_common_name = None
    if body.label:
        from app.ml.taxonomic_rollup import resolve_label_names
        from app.models.label_taxonomy import LabelTaxonomy

        tax = (
            db.query(LabelTaxonomy).get(new_taxonomy_id)
            if new_taxonomy_id
            else None
        )
        new_common_name, new_scientific_name = resolve_label_names(
            body.label, tax, body.category or ""
        )

    # When relabeling to a category-only builtin (person/vehicle/animal),
    # resolve taxonomy from the category so both names and FK are set.
    builtin_taxonomy_id = None
    builtin_scientific_name = None
    builtin_common_name = None
    if body.category and not body.label:
        from app.ml.taxonomy_db import BUILTIN_MODEL_ID
        from app.models.label_taxonomy import LabelTaxonomy

        builtin = (
            db.query(LabelTaxonomy)
            .filter(
                LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID,
                LabelTaxonomy.name == body.category,
            )
            .first()
        )
        if builtin:
            builtin_taxonomy_id = builtin.id
            builtin_scientific_name = builtin.scientific_name
            builtin_common_name = builtin.common_name

    label_provided = "label" in body.model_fields_set

    for det in detections:
        if label_provided:
            det.label = body.label if body.label else None
            det.label_confidence = 1.0 if body.label else None
            det.label_taxonomy_id = new_taxonomy_id
            det.scientific_name = new_scientific_name if body.label else None
            det.common_name = new_common_name if body.label else None
        if body.category is not None:
            det.category = body.category
        # Apply builtin taxonomy for category-only relabels
        if builtin_taxonomy_id and not det.label:
            det.label_taxonomy_id = builtin_taxonomy_id
            det.scientific_name = builtin_scientific_name
            det.common_name = builtin_common_name
        det.classification_method = "human"
        det.verified = True

    file_crud.recompute_file_verified_for_detections(db, body.detection_ids)
    db.commit()

    # Recalculate observation types for affected files. Relabel always
    # verifies the detections, and a verified box always passes, so the
    # file's observation_type can change even without a category change.
    file_ids = {det.file_id for det in detections}
    for fid in file_ids:
        file_crud.recalculate_observation_type(db, fid)

    _recalculate_max_n(db, body.detection_ids)
    return {"updated_count": len(detections)}


@router.post("/bulk-revert-to-original")
def bulk_revert_to_original(
    body: BulkRevertRequest,
    db: Session = Depends(get_db),
):
    """Undo human label edits / verifications (max 500).

    Restores each detection to the machine's final label from the
    ``original_*`` columns (the surfaced post-rollup / post-smoothing call,
    which relabeling never overwrites): label, label_confidence, taxonomy
    FK, and display names, with classification_method reset to "machine"
    (its fresh-processed value). Clears the verified flag. Category is left
    as-is — there is no original category stored, and the label actions do
    not change it.

    Powers the labels grid's Undo. Returns the reverted rows so the
    client can patch its grid in place without a re-sort.
    """
    from app.api.crud.detection import _resolve_detection_taxonomy
    from app.ml.taxonomic_rollup import resolve_label_names
    from app.models.label_taxonomy import LabelTaxonomy

    if not body.detection_ids:
        return {"reverted": []}

    detections = (
        db.query(Detection)
        .filter(Detection.id.in_(body.detection_ids))
        .all()
    )
    if not detections:
        raise HTTPException(status_code=404, detail="No detections found")

    reverted: list[dict] = []
    for det in detections:
        orig = det.original_label
        if orig:
            tax_id = _resolve_detection_taxonomy(db, det, orig)
            tax = db.query(LabelTaxonomy).get(tax_id) if tax_id else None
            common, scientific = resolve_label_names(orig, tax, det.category or "")
            det.label = orig
            det.label_confidence = det.original_label_confidence
            det.label_taxonomy_id = tax_id
            det.scientific_name = scientific
            det.common_name = common
            det.classification_method = "machine"
        else:
            det.label = None
            det.label_confidence = None
            det.label_taxonomy_id = None
            det.scientific_name = None
            det.common_name = None
            det.classification_method = None
        det.verified = False
        det.verified_at_utc = None
        reverted.append(
            {
                "detection_id": det.id,
                "label": det.label,
                "category": det.category,
                "label_confidence": det.label_confidence,
                "label_taxonomy_id": det.label_taxonomy_id,
                "scientific_name": det.scientific_name,
                "common_name": det.common_name,
                "verified": det.verified,
            }
        )

    file_crud.recompute_file_verified_for_detections(db, body.detection_ids)
    db.commit()

    # Unverifying can flip observation_type back (a box that only passed
    # because it was verified no longer counts), so recompute per file.
    file_ids = {det.file_id for det in detections}
    for fid in file_ids:
        file_crud.recalculate_observation_type(db, fid)

    _recalculate_max_n(db, body.detection_ids)
    return {"reverted": reverted}
