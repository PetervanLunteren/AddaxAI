"""
Pydantic schemas for Deployment Queue API.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- Clear separation of create/update/response schemas
"""

from datetime import datetime

from pydantic import BaseModel, Field


class DeploymentQueueBase(BaseModel):
    """Base schema with common deployment queue fields."""

    folder_path: str = Field(..., min_length=1, description="Absolute path to deployment folder")
    site_id: str | None = Field(None, description="Site ID (optional)")
    datetime_offset_seconds: int | None = Field(
        None,
        description="Seconds to add to all file timestamps (null = no adjustment)",
    )
    use_file_mtime_fallback: bool = Field(
        False,
        description=(
            "Fill missing capture dates from each file's modification time. "
            "Never overrides a real capture date"
        ),
    )
    paired_cameras: bool = Field(
        False,
        description=(
            "The subfolders are dependent cameras: cluster "
            "events across them and count effort once"
        ),
    )
    notes: str | None = Field(
        None, max_length=1000, description="Optional deployment notes"
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="Custom key:value metadata tags"
    )


class DeploymentQueueCreate(DeploymentQueueBase):
    """
    Schema for creating a new queue entry.

    folder_path is required, all other fields are optional.
    """

    project_id: str = Field(..., description="Project ID")
    video_count: int = Field(default=0, description="Number of videos in deployment folder")
    image_count: int = Field(default=0, description="Number of images in deployment folder")


class DeploymentQueueResponse(DeploymentQueueBase):
    """
    Schema for queue entry responses.

    Includes all fields plus generated id and timestamps.
    """

    id: str
    project_id: str
    video_count: int = Field(default=0, description="Number of videos in deployment folder")
    image_count: int = Field(default=0, description="Number of images in deployment folder")
    status: str = Field(..., description="Queue status: pending, processing, completed, failed")
    created_at_utc: datetime
    processed_at_utc: datetime | None = None
    error: str | None = None
    warnings: str | None = Field(
        None,
        description=(
            "Non-fatal warnings from the last ingest, newline-joined. "
            "Populated e.g. when some files were skipped because they "
            "had no extractable capture timestamp."
        ),
    )
    deployment_id: str | None = Field(None, description="Created deployment ID after processing")

    model_config = {"from_attributes": True}  # Enable ORM mode for SQLAlchemy models


class ProcessQueueRequest(BaseModel):
    """
    Schema for processing queue request.

    Specifies which project's queue to process.
    """

    project_id: str = Field(..., description="Project ID to process queue for")
