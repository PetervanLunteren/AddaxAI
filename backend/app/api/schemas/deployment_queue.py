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
    detection_model_id: str | None = Field(None, description="Detection model ID (e.g., 'megadetector_v5a')")
    classification_model_id: str | None = Field(None, description="Classification model ID (e.g., 'EUR-DF-v1-3')")
    species_list: dict | None = Field(None, description="Expected species list")


class DeploymentQueueCreate(DeploymentQueueBase):
    """
    Schema for creating a new queue entry.

    folder_path is required, all other fields are optional.
    """

    project_id: str = Field(..., description="Project ID")


class DeploymentQueueResponse(DeploymentQueueBase):
    """
    Schema for queue entry responses.

    Includes all fields plus generated id and timestamps.
    """

    id: str
    project_id: str
    status: str = Field(..., description="Queue status: pending, processing, completed, failed")
    created_at: datetime
    processed_at: datetime | None = None
    error: str | None = None
    deployment_id: str | None = Field(None, description="Created deployment ID after processing")

    model_config = {"from_attributes": True}  # Enable ORM mode for SQLAlchemy models


class ProcessQueueRequest(BaseModel):
    """
    Schema for processing queue request.

    Specifies which project's queue to process.
    """

    project_id: str = Field(..., description="Project ID to process queue for")
