"""
Undo schemas: the before-state of an action, and what a restore returns.

Every undoable action on the Labels page (verify, unverify, relabel, mark
false, mark unknown, on boxes or on files) returns an ``UndoSnapshot`` of
the rows it is about to change, taken before it changes them. The client
keeps that snapshot as its undo entry and sends it back to
``POST /api/undo``, which writes the values back verbatim. Undo therefore
puts back exactly what was there, not the model's prediction.

``label_taxonomy_id`` is deliberately not part of the snapshot. It is
re-derived from the label on restore, by the same single writer relabel
uses, because the column is ``ON DELETE SET NULL`` and a custom label row
can be gone by the time the undo arrives.
"""

from datetime import datetime

from pydantic import BaseModel

from app.api.schemas.detection import DetectionResponse
from app.models import Detection, File


class DetectionState(BaseModel):
    """The user-editable state of one box, as it was before an action."""

    detection_id: str
    category: str
    label: str | None
    label_confidence: float | None
    scientific_name: str | None
    common_name: str | None
    classification_method: str | None
    verified: bool
    verified_at_utc: datetime | None

    @classmethod
    def from_detection(cls, det: Detection) -> "DetectionState":
        return cls(
            detection_id=det.id,
            category=det.category,
            label=det.label,
            label_confidence=det.label_confidence,
            scientific_name=det.scientific_name,
            common_name=det.common_name,
            classification_method=det.classification_method,
            verified=det.verified,
            verified_at_utc=det.verified_at_utc,
        )


class FileState(BaseModel):
    """A file's sign-off, as it was before an action."""

    file_id: str
    verified: bool
    verified_at_utc: datetime | None

    @classmethod
    def from_file(cls, file: File) -> "FileState":
        return cls(
            file_id=file.id,
            verified=file.verified,
            verified_at_utc=file.verified_at_utc,
        )


class UndoSnapshot(BaseModel):
    """Everything an action changed, as it was before. Empty means nothing."""

    detections: list[DetectionState] = []
    files: list[FileState] = []


class UndoResponse(BaseModel):
    """The boxes a restore wrote, so the grid can patch them in place."""

    detections: list[DetectionResponse]
