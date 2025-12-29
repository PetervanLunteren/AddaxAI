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
    detection_model_id: str = Field(default="MD5A-0-0", description="Detection model ID")
    classification_model_id: str | None = Field(None, description="Classification model ID or null for detection-only")
    excluded_classes: list[str] = Field(default_factory=list, description="Species classes to exclude from classification")

    # SpeciesNet geographic location (alternative to excluded_classes)
    country_code: str | None = Field(None, description="ISO country code for SpeciesNet models (e.g., 'USA', 'KEN')")
    state_code: str | None = Field(None, description="US state code for SpeciesNet models (e.g., 'CA', 'TX')")

    # Detection and processing settings
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold for detections (0.0-1.0)")
    event_smoothing: bool = Field(default=True, description="Apply temporal smoothing to detections")
    taxonomic_rollup: bool = Field(default=True, description="Aggregate detections by taxonomy")
    taxonomic_rollup_threshold: float = Field(default=0.65, ge=0.1, le=1.0, description="Confidence threshold for taxonomic rollup (0.1-1.0)")
    independence_interval: int = Field(default=1800, ge=0, description="Minimum time between independent events (seconds)")


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
    excluded_classes: list[str] | None = None
    country_code: str | None = None
    state_code: str | None = None
    detection_threshold: float | None = Field(None, ge=0.0, le=1.0)
    event_smoothing: bool | None = None
    taxonomic_rollup: bool | None = None
    taxonomic_rollup_threshold: float | None = Field(None, ge=0.1, le=1.0)
    independence_interval: int | None = Field(None, ge=0)


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
