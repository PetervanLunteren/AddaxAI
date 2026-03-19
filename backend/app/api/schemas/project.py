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
    classification_model_id: str | None = Field(
        None, description="Classification model ID or null for detection-only"
    )
    embedding_model_id: str | None = Field(
        "DINOV2-VITB14", description="Embedding model ID or null to skip embeddings"
    )
    excluded_classes: list[str] = Field(
        default_factory=list, description="Species classes to exclude from classification"
    )

    # SpeciesNet geographic location (alternative to excluded_classes)
    country_code: str | None = Field(
        None, description="ISO country code for SpeciesNet models (e.g., 'USA', 'KEN')"
    )
    state_code: str | None = Field(
        None, description="US state code for SpeciesNet models (e.g., 'CA', 'TX')"
    )

    # Verification shortcut labels (keys 1-5 → label options)
    shortcut_labels: dict = Field(
        default_factory=dict,
        description="Keyboard shortcut label mappings for verification (keys 1-5)",
    )

    # Video processing settings
    video_fps: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Frames per second to extract from videos (0.1-10.0)",
    )

    # Detection and processing settings
    detection_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence threshold for detections (0.0-1.0)"
    )
    event_smoothing: bool = Field(
        default=True, description="Apply temporal smoothing to detections"
    )
    smoothing_strength: str = Field(
        default="normal",
        pattern="^(mild|normal|aggressive)$",
        description="Smoothing aggressiveness: mild, normal, or aggressive",
    )
    taxonomic_rollup: bool = Field(default=True, description="Aggregate detections by taxonomy")
    taxonomic_rollup_threshold: float = Field(
        default=0.65,
        ge=0.1,
        le=1.0,
        description="Confidence threshold for taxonomic rollup (0.1-1.0)",
    )
    independence_interval: int = Field(
        default=1800, ge=0, description="Minimum time between independent events (seconds)"
    )

    # Similarity clustering defaults (HDBSCAN params)
    min_cluster_size: int = Field(5, ge=2, le=100, description="HDBSCAN min_cluster_size parameter")
    min_samples: int = Field(3, ge=1, le=50, description="HDBSCAN min_samples parameter")


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
    embedding_model_id: str | None = None
    excluded_classes: list[str] | None = None
    country_code: str | None = None
    state_code: str | None = None
    shortcut_labels: dict | None = None
    video_fps: float | None = Field(None, ge=0.1, le=10.0)
    detection_threshold: float | None = Field(None, ge=0.0, le=1.0)
    event_smoothing: bool | None = None
    smoothing_strength: str | None = Field(None, pattern="^(mild|normal|aggressive)$")
    taxonomic_rollup: bool | None = None
    taxonomic_rollup_threshold: float | None = Field(None, ge=0.1, le=1.0)
    independence_interval: int | None = Field(None, ge=0)
    min_cluster_size: int | None = Field(None, ge=2, le=100)
    min_samples: int | None = Field(None, ge=1, le=50)


class ProjectResponse(ProjectBase):
    """
    Schema for project responses.

    Includes all fields plus generated id and timestamps.
    """

    id: str
    created_at: datetime
    updated_at: datetime
    postprocessing_settings_hash: str | None = None
    thumbnail_path: str | None = None

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
    trap_nights: int = Field(0, description="Total trap nights across all deployments")


class CustomLabelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class CustomLabelResponse(BaseModel):
    id: str
    name: str
    level: str
    taxon_class: str | None = None
    taxon_order: str | None = None
    taxon_family: str | None = None
    taxon_genus: str | None = None
    taxon_species: str | None = None

    model_config = {"from_attributes": True}


class CustomLabelUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    taxon_class: str | None = Field(None, max_length=100)
    taxon_order: str | None = Field(None, max_length=100)
    taxon_family: str | None = Field(None, max_length=100)
    taxon_genus: str | None = Field(None, max_length=100)
    taxon_species: str | None = Field(None, max_length=100)


class GBIFSuggestion(BaseModel):
    gbif_key: int
    scientific_name: str
    canonical_name: str
    rank: str
    taxon_class: str | None = None
    taxon_order: str | None = None
    taxon_family: str | None = None
    taxon_genus: str | None = None
    taxon_species: str | None = None
