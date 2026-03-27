"""
Detections API router.

Provides endpoints for creating, updating, and deleting detections
(human-drawn annotations), crop thumbnails, and detection-level verification.
"""

from datetime import datetime

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
from app.models import Detection
from app.services.crop_service import get_or_create_crop, invalidate_crop_cache

router = APIRouter(prefix="/api/detections", tags=["detections"])


def _recalculate_max_n(db: Session, detection_ids: list[str]) -> None:
    """Recalculate MaxN for events affected by the given detections."""
    event_ids = get_event_ids_for_detections(db, detection_ids)
    if not event_ids:
        return
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
    """
    detection = detection_crud.create_human_detection(db, data)
    file_crud.recalculate_observation_type(db, data.file_id)
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

    # Recalculate observation type if category changed
    if update.category is not None:
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
    threshold = get_project_threshold_for_detections(db, det_ids) if det_ids else 0.0
    count = detection_crud.delete_detections_by_file(db, file_id)
    file_crud.recalculate_observation_type(db, file_id)
    if affected_event_ids:
        recalculate_max_n_for_events(db, affected_event_ids, threshold)
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
    detection.verified_at = datetime.utcnow() if body.verified else None
    db.commit()
    db.refresh(detection)
    _recalculate_max_n(db, [detection_id])
    return detection


@router.post("/bulk-verify")
def bulk_verify_detections(
    body: BulkVerifyRequest,
    db: Session = Depends(get_db),
):
    """Bulk verify/unverify detections (max 500)."""
    now = datetime.utcnow() if body.verified else None
    updated = (
        db.query(Detection)
        .filter(Detection.id.in_(body.detection_ids))
        .update(
            {"verified": body.verified, "verified_at": now},
            synchronize_session="fetch",
        )
    )
    db.commit()
    _recalculate_max_n(db, body.detection_ids)
    return {"updated_count": updated}


@router.post("/bulk-relabel")
def bulk_relabel_detections(
    body: BulkRelabelRequest,
    db: Session = Depends(get_db),
):
    """Bulk relabel detections (max 500). Sets classification_method='human'."""
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

    # Read display_name from the taxonomy row (single source of truth)
    new_display_name = None
    if body.label and new_taxonomy_id:
        from app.models.label_taxonomy import LabelTaxonomy

        tax = db.query(LabelTaxonomy).get(new_taxonomy_id)
        new_display_name = (
            tax.display_name
            if tax
            else body.label[0].upper() + body.label[1:]
        )
    elif body.label:
        new_display_name = body.label[0].upper() + body.label[1:]

    # When relabeling to a category-only builtin (person/vehicle/animal),
    # resolve taxonomy from the category so display_name and FK are set.
    builtin_taxonomy_id = None
    builtin_display_name = None
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
            builtin_display_name = builtin.display_name

    label_provided = "label" in body.model_fields_set

    for det in detections:
        if label_provided:
            det.label = body.label if body.label else None
            det.label_confidence = 1.0 if body.label else None
            det.label_taxonomy_id = new_taxonomy_id
            det.display_name = new_display_name if body.label else None
        if body.category is not None:
            det.category = body.category
        # Apply builtin taxonomy for category-only relabels
        if builtin_taxonomy_id and not det.label:
            det.label_taxonomy_id = builtin_taxonomy_id
            det.display_name = builtin_display_name
        det.classification_method = "human"
        det.verified = True

    db.commit()

    # Recalculate observation types for affected files
    if body.category is not None:
        file_ids = {det.file_id for det in detections}
        for fid in file_ids:
            file_crud.recalculate_observation_type(db, fid)

    _recalculate_max_n(db, body.detection_ids)
    return {"updated_count": len(detections)}
