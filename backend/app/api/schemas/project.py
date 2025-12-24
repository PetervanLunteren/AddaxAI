"""
Pydantic schemas for Project API.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- Clear separation of create/update/response schemas
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ProjectBase(BaseModel):
    """Base schema with common project fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: str | None = Field(None, description="Optional project description")
    detection_model_id: str = Field(..., description="Detection model ID (e.g., 'MD5A-0-0')")
    classification_model_id: str | None = Field(None, description="Classification model ID or null for detection-only")
    taxonomy_config: dict = Field(default_factory=dict, description="Selected species classes configuration")


class ProjectCreate(ProjectBase):
    """
    Schema for creating a new project.

    Name is required, description is optional.
    """

    pass


class ProjectUpdate(BaseModel):
    """
    Schema for updating an existing project.

    All fields are optional - only provided fields will be updated.
    """

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    detection_model_id: str | None = None
    classification_model_id: str | None = None
    taxonomy_config: dict | None = None


class ProjectResponse(ProjectBase):
    """
    Schema for project responses.

    Includes all fields plus generated id and timestamps.
    """

    id: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}  # Enable ORM mode for SQLAlchemy models


class ProjectWithStats(ProjectResponse):
    """
    Extended project response with statistics.

    Used for /api/projects/{id}/stats endpoint.
    """

    site_count: int = Field(0, description="Number of sites in this project")
    deployment_count: int = Field(0, description="Number of deployments in this project")
    file_count: int = Field(0, description="Total number of files in this project")
    detection_count: int = Field(0, description="Total number of detections in this project")
