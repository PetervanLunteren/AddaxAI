"""One animal's frames, for the opened track on the Detections tab.

A card in the Labels grid stands for a whole track, and a verdict on it
reaches every box of the animal. Opening the card asks here for the
frames behind it, so a person can disagree with part of a track without
cutting it in two: relabel the frames that are wrong and every count
downstream is right, because MaxN reads what is in each frame rather
than what the tracker believed.

Not every box of the track is a frame to check. The tracker keeps boxes
down to MegaDetector's storage floor, because a weak box that overlaps
where the animal was expected is what holds a blurred or half-hidden
animal together. A person needs only the frames whose label can be
wrong, the ones the classifier looked at, so the frames are scoped at
the classification gate the analysis ran with.

The pictures are not here. Each box gets its crop from
``/api/detections/{id}/crop`` as any card does, which decodes that frame
on request. Nothing per frame is stored.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.schemas.detection import DetectionResponse
from app.api.schemas.file import TrackResponse
from app.api.schemas.label import CropBbox
from app.db.base import get_db
from app.ml.inference.crop_box import bbox_within_crop
from app.ml.label_exclusion import threshold_or_verified
from app.models import Deployment, Detection, File, Track

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
    """The animal, and the frames of it a person may check."""

    file_id: str
    file_name: str
    frame_rate: float | None
    duration_seconds: float | None
    track: TrackResponse
    detections: list[TrackDetectionRow]


@router.get("/{track_id}/detections", response_model=TrackDetectionsResponse)
def get_track_detections(
    track_id: str,
    db: Session = Depends(get_db),
) -> TrackDetectionsResponse:
    """The track's frames at or above the run's classification gate, or
    verified, in frame order.

    The line is the gate the deployment was analysed with
    (``classification_gate_used``), so it is fixed for the life of the
    analysis. Not the counting threshold, which Refine results moves after
    the fact, and not the Labels slider, which filters cards: neither may
    redefine what is inside a track. A frame below the gate never got a
    species guess of its own and sits far below the default counting
    threshold, so there is nothing in it to check. It stays stored and
    keeps holding the track together.

    Never empty for a card on screen: a track starts from a box at the
    default counting threshold, far above any gate in use, and its card is
    its best box.
    """
    track = db.query(Track).filter(Track.id == track_id).first()
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    file = db.query(File).filter(File.id == track.file_id).first()
    if file is None:
        raise HTTPException(status_code=404, detail="File not found")

    gate = (
        db.query(Deployment.classification_gate_used)
        .filter(Deployment.id == file.deployment_id)
        .scalar()
    )
    if gate is None:
        # Every analysis since 2026-07 records its gate, and tracks came
        # later, so a track without one means a broken record. Say so.
        raise ValueError(
            f"Deployment {file.deployment_id} has a track but no recorded "
            "classification gate. Refusing to guess which frames to show."
        )

    rows = (
        db.query(Detection)
        .filter(Detection.track_id == track_id, threshold_or_verified(gate))
        .order_by(Detection.frame_number, Detection.id)
        .all()
    )

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
