"""One animal's frames, for the opened track on the Detections tab.

A card in the Labels grid stands for a whole track, and a verdict on it
reaches every box of the animal. Opening the card asks here for the
boxes behind it, so a person can disagree with part of a track without
cutting it in two: relabel the frames that are wrong and every count
downstream is right, because MaxN reads what is in each frame rather
than what the tracker believed.

The pictures are not here. Each box gets its crop from
``/api/detections/{id}/crop`` as any card does, which decodes that frame
on request. Nothing per frame is stored.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.crud.file import _project_threshold_for_file
from app.api.schemas.detection import DetectionResponse
from app.api.schemas.file import TrackResponse
from app.api.schemas.label import CropBbox
from app.core.confidence import effective_floor
from app.db.base import get_db
from app.ml.inference.crop_box import bbox_within_crop
from app.ml.label_exclusion import threshold_or_verified
from app.models import Detection, File, Track

router = APIRouter(prefix="/api/tracks", tags=["tracks"])


class TrackDetectionRow(DetectionResponse):
    """One frame of the track, as a card.

    ``DetectionResponse`` plus where the box sits inside its finished
    crop, which the card draws its overlay from. The grid gets the same
    number from ``similarity_script``; both call
    ``crop_box.bbox_within_crop``, so a card built here and a card built
    there cannot disagree.
    """

    crop_bbox: CropBbox | None = None


class TrackDetectionsResponse(BaseModel):
    """The animal, and the frames of it a person may act on."""

    file_id: str
    file_name: str
    frame_rate: float | None
    duration_seconds: float | None
    track: TrackResponse
    detections: list[TrackDetectionRow]


@router.get("/{track_id}/detections", response_model=TrackDetectionsResponse)
def get_track_detections(
    track_id: str,
    min_confidence: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="The Labels page confidence slider, if the user moved it.",
    ),
    db: Session = Depends(get_db),
) -> TrackDetectionsResponse:
    """The track's boxes, in frame order, at the grid's confidence scope.

    The scope is the grid's, not the project's alone: the slider digs
    *down* below the counting threshold (`effective_floor`), so a card
    visible at a lowered slider must open to the frames that were
    visible with it. Gating at the project threshold instead would open
    an empty track under a card the person can see, which is the one
    failure that makes the feature look broken rather than slow.

    Returns fewer rows than ``track.frame_count``: the tracker keeps
    boxes down to the storage floor while the grid shows from the
    counting threshold. The header above the grid says so; the number
    on the badge is how many frames the animal was followed for, which
    is a fact about the animal rather than a promise about the grid.
    """
    track = db.query(Track).filter(Track.id == track_id).first()
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    file = db.query(File).filter(File.id == track.file_id).first()
    if file is None:
        raise HTTPException(status_code=404, detail="File not found")

    floor = effective_floor(_project_threshold_for_file(db, file), min_confidence)
    query = (
        db.query(Detection)
        .filter(Detection.track_id == track_id)
        .filter(threshold_or_verified(floor))
    )
    if min_confidence is not None:
        # The slider is also applied literally, as the grid applies it: a
        # verified low-confidence box passes the floor's OR clause but
        # cannot satisfy a min the user set by hand.
        query = query.filter(Detection.confidence >= min_confidence)
    rows = query.order_by(Detection.frame_number, Detection.id).all()

    return TrackDetectionsResponse(
        file_id=file.id,
        file_name=file.file_path.rsplit("/", 1)[-1] if file.file_path else "",
        frame_rate=file.frame_rate,
        duration_seconds=file.duration_seconds,
        track=TrackResponse.model_validate(track, from_attributes=True),
        detections=[
            TrackDetectionRow(
                **DetectionResponse.model_validate(
                    det, from_attributes=True
                ).model_dump(),
                crop_bbox=bbox_within_crop(
                    det.bbox_width, det.bbox_height, file.width_px, file.height_px
                ),
            )
            for det in rows
            if det.bbox_width is not None and det.bbox_height is not None
        ],
    )
