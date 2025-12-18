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

    class Config:
        from_attributes = True


class FileWithDetections(FileResponse):
    """File with detections response schema."""

    detections: list[DetectionResponse]

    class Config:
        from_attributes = True
