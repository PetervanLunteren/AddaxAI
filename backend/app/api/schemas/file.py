"""
File schemas for API requests and responses.
"""

from datetime import datetime

from pydantic import BaseModel


class DetectionResponse(BaseModel):
    """Detection response schema."""

    id: str
    category: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    species: str | None
    species_confidence: float | None
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
    timestamp: datetime
    created_at: datetime
    best_frame_number: int | None = None
    best_frame_path: str | None = None
    frame_rate: float | None = None
    observation_type: str = "unclassified"
    verified: bool = False
    verified_at: datetime | None = None
    notes: str | None = None
    favorited: bool = False
    source_video_id: str | None = None
    source_frame_number: int | None = None

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
