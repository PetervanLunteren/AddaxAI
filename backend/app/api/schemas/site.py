"""
Pydantic schemas for Site API.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- GPS coordinate validation
"""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


def reject_null_island(latitude: float | None, longitude: float | None) -> None:
    """Reject 0,0: it is the form default and 'null island', so almost always
    a forgotten location rather than a real site."""
    if latitude == 0 and longitude == 0:
        raise ValueError(
            "0, 0 is not allowed. This is probably an error. Enter the real coordinates."
        )


class SiteBase(BaseModel):
    """Base schema with common site fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Site name")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    elevation_m: float | None = Field(None, description="Elevation in meters")
    habitat_type: str | None = Field(None, max_length=255, description="Habitat type")
    notes: str | None = Field(None, max_length=1000, description="Additional notes")
    tags: dict[str, str] = Field(
        default_factory=dict, description="Custom key:value metadata tags"
    )

    # NOTE: the 0,0 null-island check lives on the INPUT schemas (SiteCreate,
    # SiteUpdate), not here. SiteResponse also extends SiteBase, and putting the
    # validator here made reading back an existing 0,0 site fail with a 500.


class SiteCreate(SiteBase):
    """
    Schema for creating a new site.

    Requires project_id to associate site with project.
    """

    project_id: str = Field(..., description="ID of the project this site belongs to")

    @model_validator(mode="after")
    def _check_null_island(self) -> "SiteCreate":
        reject_null_island(self.latitude, self.longitude)
        return self


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
    notes: str | None = Field(None, max_length=1000)
    tags: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check_null_island(self) -> "SiteUpdate":
        reject_null_island(self.latitude, self.longitude)
        return self


class SiteResponse(SiteBase):
    """
    Schema for site responses.

    Includes all fields plus generated id, project_id, and timestamps.
    """

    id: str
    project_id: str
    created_at_utc: datetime

    model_config = {"from_attributes": True}  # Enable ORM mode for SQLAlchemy models


class SiteWithStats(SiteResponse):
    """Site response with deployment count for table views."""

    deployment_count: int = 0


class SiteFileCounts(BaseModel):
    """File-type breakdown for a single site."""

    total: int
    images: int
    videos: int


class SiteTopSpecies(BaseModel):
    """One row in the top-species leaderboard for a site."""

    label: str
    common_name: str | None
    scientific_name: str | None
    count: int


class SiteDetectionCategories(BaseModel):
    """Observation counts split by detection category + blank files."""

    animal: int
    person: int
    vehicle: int
    empty: int


class SiteVerification(BaseModel):
    """File verification progress aggregated across the site's deployments."""

    verified: int
    total: int


class SiteInfoResponse(BaseModel):
    """
    Investigation-level payload for the Sites → Info sheet.

    Aggregates across every deployment at this site: file and size
    totals, verification progress, event and observation counts, the
    detection-category breakdown, the top species, trap nights and the
    observation rate, plus the first and last capture timestamps.
    """

    site_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float | None
    habitat_type: str | None
    notes: str | None
    tags: dict[str, str]
    deployment_count: int
    files: SiteFileCounts
    # Sum of File.size_bytes across the site's files. 0 when no files
    # or when size_bytes is null for every file.
    total_size_bytes: int
    verification: SiteVerification
    event_count: int
    observation_count: int
    detection_categories: SiteDetectionCategories
    top_species: list[SiteTopSpecies]
    # Sum of per-deployment (end - start + 1) days. None when the site
    # has no deployments or any deployment is open-ended (no end_date).
    trap_nights: int | None
    observation_rate_per_100_trap_nights: float | None
    first_captured_at_local: datetime | None
    last_captured_at_local: datetime | None
