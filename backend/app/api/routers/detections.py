"""
Detections API router.

Provides endpoints for creating, updating, and deleting detections
(human-drawn annotations).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.crud import detection as detection_crud
from app.api.crud import file as file_crud
from app.api.schemas.detection import (
    DetectionCreateHuman,
    DetectionResponse,
    DetectionUpdate,
)
from app.db.base import get_db

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
    """
    detection = detection_crud.update_detection(db, detection_id, update)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    # Recalculate observation type if category changed
    if update.category is not None:
        file_crud.recalculate_observation_type(db, detection.file_id)

    return detection


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
