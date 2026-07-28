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
    """Base schema with common detection fields.

    Bounding box fields are nullable to accommodate event-level
    observations (a user noting "deer seen in this clip" without a
    frame-anchored ROI). The four coordinates must be all-set or
    all-null per row; the `validate_bbox_all_or_nothing` validator
    enforces this. Aligns with Camtrap-DP, which makes bboxX/Y/W/H
    explicitly optional and supports observationLevel="event".
    """

    category: DetectionCategory = Field(..., description="Detection category")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence (0.0-1.0)"
    )
    bbox_x: float | None = Field(
        None, ge=0.0, le=1.0, description="Bounding box top-left X (normalized 0-1)"
    )
    bbox_y: float | None = Field(
        None, ge=0.0, le=1.0, description="Bounding box top-left Y (normalized 0-1)"
    )
    bbox_width: float | None = Field(
        None, ge=0.0, le=1.0, description="Bounding box width (normalized 0-1)"
    )
    bbox_height: float | None = Field(
        None, ge=0.0, le=1.0, description="Bounding box height (normalized 0-1)"
    )
    label: str | None = Field(None, max_length=100, description="Label name")
    label_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Label classification confidence"
    )
    common_name: str | None = Field(
        None, max_length=100, description="Common name (or cleaned class label)"
    )
    scientific_name: str | None = Field(
        None, max_length=100, description="Latin taxonomy display name"
    )
    label_taxonomy_id: str | None = Field(
        None, description="FK to label_taxonomy table"
    )
    classification_method: str | None = Field(
        None, description="Classification method: 'machine' or 'human'"
    )
    frame_number: int | None = Field(
        None, ge=0, description="Frame number for video detections (None for images)"
    )

    @field_validator("label_confidence")
    @classmethod
    def validate_label_confidence(cls, v: float | None, info) -> float | None:
        """Ensure label_confidence is only set if label is provided."""
        if v is not None and info.data.get("label") is None:
            raise ValueError("label_confidence requires label to be set")
        return v

    @field_validator("bbox_height")
    @classmethod
    def validate_bbox_all_or_nothing(cls, v: float | None, info) -> float | None:
        """All four bbox coordinates must be set together, or all null.
        Runs on the last bbox field so the other three are in `info.data`."""
        present = [
            info.data.get("bbox_x") is not None,
            info.data.get("bbox_y") is not None,
            info.data.get("bbox_width") is not None,
            v is not None,
        ]
        if any(present) and not all(present):
            raise ValueError(
                "bbox_x, bbox_y, bbox_width, bbox_height must all be set "
                "or all be null"
            )
        return v


class DetectionCreate(DetectionBase):
    """
    Schema for creating a new detection.

    Requires file_id and job_id to track which file and job created this detection.
    """

    file_id: str = Field(..., description="ID of the file this detection belongs to")
    job_id: str = Field(..., description="ID of the job that created this detection")


class DetectionCreateHuman(BaseModel):
    """Schema for creating a human-drawn detection (no job_id required)."""

    file_id: str = Field(..., description="ID of the file")
    category: DetectionCategory = Field(..., description="Detection category")
    bbox_x: float = Field(..., ge=0.0, le=1.0)
    bbox_y: float = Field(..., ge=0.0, le=1.0)
    bbox_width: float = Field(..., ge=0.0, le=1.0)
    bbox_height: float = Field(..., ge=0.0, le=1.0)
    label: str | None = Field(None, max_length=100)
    # Anchors the new box to a video frame so the overlay still
    # renders it. Null for images.
    frame_number: int | None = Field(None, ge=0)


class DetectionUpdate(BaseModel):
    """Schema for updating a detection (all fields optional)."""

    category: DetectionCategory | None = None
    bbox_x: float | None = Field(None, ge=0.0, le=1.0)
    bbox_y: float | None = Field(None, ge=0.0, le=1.0)
    bbox_width: float | None = Field(None, ge=0.0, le=1.0)
    bbox_height: float | None = Field(None, ge=0.0, le=1.0)
    label: str | None = None
    label_confidence: float | None = Field(None, ge=0.0, le=1.0)


class DetectionResponse(DetectionBase):
    """
    Schema for detection responses.

    Includes all fields plus generated id and timestamp.
    """

    id: str
    file_id: str
    job_id: str | None
    verified: bool = False
    verified_at_utc: datetime | None = None
    created_at_utc: datetime

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
