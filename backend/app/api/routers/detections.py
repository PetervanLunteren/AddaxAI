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
from app.api.schemas.detection import (
    DetectionCreateHuman,
    DetectionResponse,
    DetectionUpdate,
)
from app.db.base import get_db
from app.models import Detection
from app.services.crop_service import get_or_create_crop, invalidate_crop_cache

router = APIRouter(prefix="/api/detections", tags=["detections"])


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
    return detection


@router.patch("/{detection_id}", response_model=DetectionResponse)
def update_detection(
    detection_id: str,
    update: DetectionUpdate,
    db: Session = Depends(get_db),
):
    """
    Update a detection's category, bounding box, or species.

    Sets classification_method="human" when species is edited.
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

    return detection


@router.delete("/by-file/{file_id}")
def delete_detections_by_file(
    file_id: str,
    db: Session = Depends(get_db),
):
    """Delete all detections for a file."""
    count = detection_crud.delete_detections_by_file(db, file_id)
    file_crud.recalculate_observation_type(db, file_id)
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
    detection_crud.delete_detection(db, detection_id)
    file_crud.recalculate_observation_type(db, file_id)


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
    species: str | None = None
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

    # Resolve taxonomy ID for the new species label
    new_taxonomy_id = None
    if body.species:
        from app.api.crud.detection import _resolve_detection_taxonomy
        new_taxonomy_id = _resolve_detection_taxonomy(db, detections[0], body.species)

    for det in detections:
        if body.species is not None:
            det.species = body.species if body.species != "" else None
            det.species_confidence = 1.0 if body.species else None
            det.species_taxonomy_id = new_taxonomy_id
        if body.category is not None:
            det.category = body.category
        det.classification_method = "human"
        det.verified = True

    db.commit()

    # Recalculate observation types for affected files
    if body.category is not None:
        file_ids = {det.file_id for det in detections}
        for fid in file_ids:
            file_crud.recalculate_observation_type(db, fid)

    return {"updated_count": len(detections)}
