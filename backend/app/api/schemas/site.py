"""
Pydantic schemas for Site API.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- GPS coordinate validation
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SiteBase(BaseModel):
    """Base schema with common site fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Site name")
    latitude: float | None = Field(None, ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float | None = Field(None, ge=-180, le=180, description="Longitude (-180 to 180)")
    elevation_m: float | None = Field(None, description="Elevation in meters")
    habitat_type: str | None = Field(None, max_length=255, description="Habitat type")
    notes: str | None = Field(None, description="Additional notes")


class SiteCreate(SiteBase):
    """
    Schema for creating a new site.

    Requires project_id to associate site with project.
    """

    project_id: str = Field(..., description="ID of the project this site belongs to")


class SiteUpdate(BaseModel):
    """
    Schema for updating an existing site.

    All fields are optional - only provided fields will be updated.
    Cannot change project_id after creation.
    """

    name: str | None = Field(None, min_length=1, max_length=255)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    elevation_m: float | None = None
    habitat_type: str | None = Field(None, max_length=255)
    notes: str | None = None


class SiteResponse(SiteBase):
    """
    Schema for site responses.

    Includes all fields plus generated id, project_id, and timestamps.
    """

    id: str
    project_id: str
    created_at: datetime

    model_config = {"from_attributes": True}  # Enable ORM mode for SQLAlchemy models
