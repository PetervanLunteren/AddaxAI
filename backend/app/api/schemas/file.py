"""
File schemas for API requests and responses.
"""

from datetime import datetime

from pydantic import BaseModel, field_serializer

from app.utils.datetime_serialization import serialize_local_datetime


class DetectionResponse(BaseModel):
    """Detection response schema."""

    id: str
    category: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    label: str | None
    label_confidence: float | None
    display_name: str | None = None
    label_taxonomy_id: str | None = None
    classification_method: str | None = None
    frame_number: int | None = None

    class Config:
        from_attributes = True


class FileResponse(BaseModel):
    """File response schema."""

    id: str
    deployment_id: str
    file_path: str
    file_type: str
    file_format: str
    size_bytes: int | None
    width_px: int | None
    height_px: int | None
    captured_at_local: datetime
    created_at_utc: datetime
    best_frame_number: int | None = None
    best_frame_path: str | None = None
    frame_rate: float | None = None
    observation_type: str = "unclassified"
    verified: bool = False
    verified_at_utc: datetime | None = None
    notes: str | None = None
    favorited: bool = False
    source_video_id: str | None = None
    source_frame_number: int | None = None

    # captured_at_local is naive wall-clock time at the camera. Rendered
    # with the offset that applies on the file's local date, read from
    # the active project timezone in the request context. See
    # DEVELOPERS.md "Datetime conventions".
    @field_serializer("captured_at_local")
    def _serialize_captured_at_local(self, value: datetime) -> str:
        return serialize_local_datetime(value)  # type: ignore[return-value]

    class Config:
        from_attributes = True


class FileWithDetections(FileResponse):
    """File with detections response schema."""

    detections: list[DetectionResponse]

    class Config:
        from_attributes = True


class FileUpdate(BaseModel):
    """Schema for updating a file (verification, notes, favorited)."""

    verified: bool | None = None
    notes: str | None = None
    favorited: bool | None = None
