"""
Undo API router: one endpoint that puts a before-state back.

The verification endpoints return the before-state of what they changed
(``before`` in their response). The client sends it here to undo.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.crud import undo as undo_crud
from app.api.schemas.detection import DetectionResponse
from app.api.schemas.undo import UndoResponse, UndoSnapshot
from app.db.base import get_db

router = APIRouter(prefix="/api/undo", tags=["undo"])


@router.post("", response_model=UndoResponse)
def undo(
    snapshot: UndoSnapshot,
    db: Session = Depends(get_db),
):
    """Restore the rows in the snapshot to the values it holds.

    Same shape as the bulk endpoints: an empty snapshot is a no-op, one
    that matches nothing is a 404. Rows that are gone are skipped, so the
    response can be shorter than the snapshot; the client says so.
    """
    if not snapshot.detections and not snapshot.files:
        return UndoResponse(detections=[])
    restored = undo_crud.restore(db, snapshot)
    if not restored.detections and not restored.file_count:
        raise HTTPException(status_code=404, detail="Nothing left to restore")
    return UndoResponse(
        detections=[DetectionResponse.model_validate(d) for d in restored.detections]
    )
