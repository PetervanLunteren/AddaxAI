"""
Pydantic schemas for Detection API.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- Clear separation of create/update/response schemas
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


DetectionCategory = Literal["animal", "person", "vehicle"]


class DetectionBase(BaseModel):
    """Base schema with common detection fields."""

    category: DetectionCategory = Field(..., description="Detection category")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence (0.0-1.0)"
    )
    bbox_x: float = Field(
        ..., ge=0.0, le=1.0, description="Bounding box top-left X (normalized 0-1)"
    )
    bbox_y: float = Field(
        ..., ge=0.0, le=1.0, description="Bounding box top-left Y (normalized 0-1)"
    )
    bbox_width: float = Field(
        ..., ge=0.0, le=1.0, description="Bounding box width (normalized 0-1)"
    )
    bbox_height: float = Field(
        ..., ge=0.0, le=1.0, description="Bounding box height (normalized 0-1)"
    )
    species: str | None = Field(None, max_length=100, description="Species name")
    species_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Species classification confidence"
    )

    @field_validator("species_confidence")
    @classmethod
    def validate_species_confidence(cls, v: float | None, info) -> float | None:
        """Ensure species_confidence is only set if species is provided."""
        if v is not None and info.data.get("species") is None:
            raise ValueError("species_confidence requires species to be set")
        return v


class DetectionCreate(DetectionBase):
    """
    Schema for creating a new detection.

    Requires file_id and job_id to track which file and job created this detection.
    """

    file_id: str = Field(..., description="ID of the file this detection belongs to")
    job_id: str = Field(..., description="ID of the job that created this detection")


class DetectionResponse(DetectionBase):
    """
    Schema for detection responses.

    Includes all fields plus generated id and timestamp.
    """

    id: str
    file_id: str
    job_id: str
    created_at: datetime

    model_config = {"from_attributes": True}  # Enable ORM mode for SQLAlchemy models


class DetectionStatsResponse(BaseModel):
    """
    Detection statistics response.

    Used for job and file summaries.
    """

    total: int = Field(0, description="Total number of detections")
    animal: int = Field(0, description="Number of animal detections")
    person: int = Field(0, description="Number of person detections")
    vehicle: int = Field(0, description="Number of vehicle detections")


class BoundingBox(BaseModel):
    """
    Bounding box coordinates (normalized 0-1).

    Used for frontend rendering and MegaDetector API format.
    """

    x: float = Field(..., ge=0.0, le=1.0, description="Top-left X coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Top-left Y coordinate")
    width: float = Field(..., ge=0.0, le=1.0, description="Box width")
    height: float = Field(..., ge=0.0, le=1.0, description="Box height")


class DetectionResult(BaseModel):
    """
    Detection result from ML model (MegaDetector format).

    This matches the format returned by MegaDetector CLI output.
    """

    category: DetectionCategory = Field(..., description="Detection category")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x, y, width, height] (normalized 0-1)",
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        """Validate bbox coordinates are in valid range."""
        if len(v) != 4:
            raise ValueError("bbox must contain exactly 4 values")
        for coord in v:
            if not 0.0 <= coord <= 1.0:
                raise ValueError(f"bbox coordinate {coord} must be between 0 and 1")
        return v
