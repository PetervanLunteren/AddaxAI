"""Calibration router: per-label score curves from "Is this a <label>?" answers."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.crud import calibration as calibration_crud
from app.api.schemas.calibration import (
    CalibrationAnswerCreate,
    CalibrationResponse,
    CalibrationSampleItem,
)
from app.db.base import get_db

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


@router.get("", response_model=CalibrationResponse)
def calibration_overview(
    project_id: str = Query(...),
    db: Session = Depends(get_db),
) -> CalibrationResponse:
    """One row per AI label: answers so far, thresholds and the curve."""
    try:
        return calibration_crud.get_calibration(db, project_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@router.get("/sample", response_model=list[CalibrationSampleItem])
def calibration_sample(
    project_id: str = Query(...),
    label: str = Query(...),
    db: Session = Depends(get_db),
) -> list[CalibrationSampleItem]:
    """The next batch of unanswered detections of `label`, spread over the scores."""
    try:
        return calibration_crud.draw_batch(db, project_id, label)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@router.post("/answers", status_code=204)
def calibration_answer(
    body: CalibrationAnswerCreate,
    db: Session = Depends(get_db),
) -> None:
    """Store the answer for one detection. Answering again replaces it."""
    try:
        calibration_crud.save_answer(db, body.detection_id, body.answer)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
