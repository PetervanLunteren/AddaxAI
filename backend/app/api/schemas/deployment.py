"""
Pydantic schemas for Deployment API.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- Clear separation of create/update/response schemas
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


FolderStatus = Literal["valid", "missing", "needs_relink"]


class DeploymentBase(BaseModel):
    """Base schema with common deployment fields."""

    folder_path: str | None = Field(
        None, description="Absolute path to deployment folder"
    )
    start_date: date = Field(..., description="Deployment start date")
    end_date: date | None = Field(None, description="Optional deployment end date")
    camera_model: str | None = Field(None, max_length=255, description="Camera model")
    camera_serial: str | None = Field(
        None, max_length=255, description="Camera serial number"
    )
    notes: str | None = Field(None, description="Optional notes about deployment")


class DeploymentCreate(DeploymentBase):
    """
    Schema for creating a new deployment.

    Requires site_id and start_date. folder_path should be provided
    to enable file scanning.
    """

    site_id: str = Field(..., description="ID of the site for this deployment")


class DeploymentUpdate(BaseModel):
    """
    Schema for updating an existing deployment.

    All fields are optional - only provided fields will be updated.
    Used for updating metadata and re-linking folder paths.
    """

    folder_path: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    camera_model: str | None = None
    camera_serial: str | None = None
    notes: str | None = None


class DeploymentResponse(DeploymentBase):
    """
    Schema for deployment responses.

    Includes all fields plus generated id, folder_status, and timestamps.
    """

    id: str
    site_id: str
    folder_status: FolderStatus = Field(
        "valid", description="Status of the folder path"
    )
    last_validated_at: datetime | None = Field(
        None, description="When folder was last validated"
    )
    created_at: datetime

    model_config = {"from_attributes": True}  # Enable ORM mode for SQLAlchemy models


class DeploymentWithStats(DeploymentResponse):
    """
    Extended deployment response with statistics.

    Used for detailed deployment views.
    """

    file_count: int = Field(0, description="Number of files in this deployment")
    event_count: int = Field(0, description="Number of events in this deployment")
    detection_count: int = Field(
        0, description="Total number of detections in this deployment"
    )


class GPSCoordinates(BaseModel):
    """GPS coordinates from EXIF data."""

    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")


class FolderPreviewResponse(BaseModel):
    """
    Preview of a deployment folder before running analysis.

    Provides quick counts and GPS location check without storing files in DB.
    """

    image_count: int = Field(..., description="Number of image files found")
    video_count: int = Field(..., description="Number of video files found")
    total_count: int = Field(..., description="Total number of media files")
    gps_location: GPSCoordinates | None = Field(
        None, description="Average GPS coordinates if found in EXIF"
    )
    suggested_site_id: str | None = Field(
        None, description="ID of nearby site if GPS matched"
    )
    sample_files: list[str] = Field(
        [], description="Sample of file paths (relative to deployment folder)"
    )
