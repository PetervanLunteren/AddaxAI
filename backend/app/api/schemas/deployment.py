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

FolderStatus = Literal["valid", "needs_relink"]


class DeploymentBase(BaseModel):
    """Base schema with common deployment fields."""

    folder_path: str | None = Field(
        None, description="Absolute path to deployment folder"
    )
    start_date_local: date = Field(..., description="Deployment start date (camera local)")
    end_date_local: date | None = Field(
        None, description="Optional deployment end date (camera local)"
    )
    camera_model: str | None = Field(None, max_length=255, description="Camera model")
    camera_serial: str | None = Field(
        None, max_length=255, description="Camera serial number"
    )
    notes: str | None = Field(
        None, max_length=1000, description="Optional notes about deployment"
    )
    datetime_offset_seconds: int | None = Field(
        None, description="Datetime offset in seconds applied during analysis"
    )
    paired_cameras: bool = Field(
        False,
        description=(
            "The subfolders are dependent cameras: cluster "
            "events across them and count effort once"
        ),
    )
    camera_offsets: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-camera seconds added on top of datetime_offset_seconds, "
            "keyed by subfolder name. Paired cameras only"
        ),
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="Custom key:value metadata tags"
    )


class DeploymentCreate(DeploymentBase):
    """
    Schema for creating a new deployment.

    Requires project_id and start_date. site_id is optional (users can
    run deployment-agnostic batches where the location is unknown or
    spans multiple sites). folder_path should be provided to enable
    file scanning.
    """

    project_id: str = Field(..., description="ID of the project this deployment belongs to")
    site_id: str | None = Field(
        None,
        description="ID of the site for this deployment; null if unknown or spans sites",
    )


class DeploymentUpdate(BaseModel):
    """
    Schema for updating an existing deployment.

    All fields are optional - only provided fields will be updated.
    Used for updating metadata, re-linking folder paths, and
    moving deployments between sites in the same project.
    """

    site_id: str | None = None
    folder_path: str | None = None
    start_date_local: date | None = None
    end_date_local: date | None = None
    camera_model: str | None = None
    camera_serial: str | None = None
    notes: str | None = Field(None, max_length=1000)
    datetime_offset_seconds: int | None = None
    paired_cameras: bool | None = None
    camera_offsets: dict[str, int] | None = None
    tags: dict[str, str] | None = None


class DeploymentResponse(DeploymentBase):
    """
    Schema for deployment responses.

    Includes all fields plus generated id, folder_status, and timestamps.
    """

    id: str
    project_id: str
    site_id: str | None
    folder_status: FolderStatus = Field(
        "valid", description="Status of the folder path"
    )
    last_validated_at_utc: datetime | None = Field(
        None, description="When folder was last validated (UTC)"
    )
    created_at_utc: datetime
    # Non-fatal issues recorded during analysis (files dropped for
    # missing EXIF, videos the decoder rejected, etc). Survives queue
    # cleanup. None when the run had nothing to flag.
    warnings: list[dict] | None = None

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


class DeploymentStatsOnly(BaseModel):
    """Lightweight stats for bulk deployment listing (no deployment fields)."""

    file_count: int = 0
    event_count: int = 0
    detection_count: int = 0


class DeploymentFileCounts(BaseModel):
    """File-type breakdown for a single deployment."""

    total: int
    images: int
    videos: int


class DeploymentTopSpecies(BaseModel):
    """One row in the top-species leaderboard for a deployment."""

    label: str
    # Optional name overrides from label_taxonomy (e.g. "B. taurus").
    common_name: str | None
    scientific_name: str | None
    count: int


class DeploymentDetectionCategories(BaseModel):
    """Observation counts split by detection category + blank files.

    Animal / person / vehicle are MaxN sums over events (matches the
    dashboard's `get_detection_categories`). Empty is the count of
    files whose `observation_type == "blank"` (no usable detections).
    """

    animal: int
    person: int
    vehicle: int
    empty: int


class DeploymentVerification(BaseModel):
    """File verification progress for a deployment."""

    verified: int
    total: int


class DeploymentInfoResponse(BaseModel):
    """
    Investigation-level payload for the Deployments → Info sheet.

    Read-only snapshot combining deployment metadata, file-type split,
    event and observation counts, mean detection and classification
    confidence (respecting the project's detection threshold with the
    verified override), storage size, verification progress, the top
    species, detection-category breakdown, trap nights, observation
    rate, and the first and last capture timestamps.
    """

    deployment_id: str
    folder_path: str | None
    paired_cameras: bool
    site_id: str | None
    site_name: str | None
    start_date_local: date
    end_date_local: date | None
    files: DeploymentFileCounts
    # Sum of File.size_bytes across files in this deployment. 0 when no
    # files or when size_bytes is null for every file.
    total_size_bytes: int
    verification: DeploymentVerification
    event_count: int
    # Sum of EventObservation.effective_count (human_count when set, else
    # the AI max_n) across all events in this deployment.
    observation_count: int
    detection_categories: DeploymentDetectionCategories
    # Top 5 species by observation count within this deployment. Empty
    # list when the deployment has no observations yet.
    top_species: list[DeploymentTopSpecies]
    # Subfolder-aware effort, see `app.api.crud.trap_nights`. None when
    # the deployment has no dated files.
    trap_nights: int | None
    # observation_count / trap_nights * 100. None when trap_nights is
    # None or 0.
    observation_rate_per_100_trap_nights: float | None
    # None when no detections pass the threshold-with-verified filter.
    mean_detection_confidence: float | None
    # None when no detection has a classification label.
    mean_classification_confidence: float | None
    # Earliest and latest File.captured_at_local in the deployment.
    first_captured_at_local: datetime | None
    last_captured_at_local: datetime | None
    # Non-fatal issues recorded during this deployment's analysis run.
    # Each entry is one file the pipeline skipped:
    #   {"type": "missing_timestamp", "path": ...} for files without an
    #   extractable EXIF / QuickTime capture timestamp, and
    #   {"type": "video_processing_failure", "path": ..., "reason": ...}
    #   for videos the decoder rejected (corrupt file, unsupported
    #   codec, etc). None when the run had nothing to flag.
    warnings: list[dict] | None = None


class BulkRelinkItem(BaseModel):
    """Single deployment relink instruction."""

    deployment_id: str = Field(..., description="Deployment to relink")
    new_folder_path: str = Field(
        ..., min_length=1, description="New absolute folder path"
    )


class BulkRelinkRequest(BaseModel):
    """Request body for bulk relinking multiple deployments at once."""

    replacements: list[BulkRelinkItem] = Field(
        ..., description="List of deployment relink instructions"
    )


class BulkRelinkResultItem(BaseModel):
    """Result for a single deployment in a bulk relink operation."""

    deployment_id: str
    success: bool
    files_rewritten: int = 0
    mismatches: list[str] = Field(default_factory=list)


class BulkRelinkResponse(BaseModel):
    """Per-deployment outcomes of a bulk relink operation."""

    results: list[BulkRelinkResultItem]


class SuggestRelinkTargetRequest(BaseModel):
    """Request body for the relink auto-suggest endpoint."""

    missing_path: str = Field(
        ..., min_length=1, description="Broken folder path to find a replacement for"
    )


class SuggestRelinkTargetResponse(BaseModel):
    """
    Suggested replacement for a missing folder.

    `existing_parent` is the deepest ancestor of `missing_path` that still
    exists on disk. `candidates` lists sibling directories under that
    parent ranked by name similarity to the missing basename.
    `suggested_path` is the top candidate if its similarity is high
    enough to auto-fill, otherwise null.
    """

    existing_parent: str | None = None
    suggested_path: str | None = None
    candidates: list[str] = Field(default_factory=list)


class GroupBrokenItem(BaseModel):
    """Single broken deployment sent to the group-broken endpoint."""

    id: str
    folder_path: str


class GroupBrokenRequest(BaseModel):
    """Request body for the group-broken endpoint."""

    items: list[GroupBrokenItem]


class GroupBrokenGroup(BaseModel):
    """
    One bucket of broken deployments that share a deepest-missing-ancestor.

    `prefix` is that ancestor (the path to substitute out). `suggested_path`
    is the auto-suggested replacement for `prefix` (same parent, renamed
    sibling) if one crosses the similarity threshold, otherwise null.
    """

    prefix: str
    existing_parent: str | None = None
    suggested_path: str | None = None
    items: list[GroupBrokenItem]


class GroupBrokenResponse(BaseModel):
    """Grouped broken deployments with per-group auto-suggestions."""

    groups: list[GroupBrokenGroup]


class GPSCoordinates(BaseModel):
    """GPS coordinates from EXIF data."""

    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")


class SampleFile(BaseModel):
    """A sample file from the deployment folder with its extracted datetime."""

    path: str = Field(..., description="File path relative to the deployment folder")
    file_datetime: str | None = Field(
        None, description="Extracted EXIF/metadata datetime (ISO string), or null"
    )


class SplitPreviewTarget(BaseModel):
    """One prospective child deployment in a split preview."""

    folder_path: str = Field(..., description="Absolute path to the child subfolder")
    name: str = Field(..., description="Display name derived from the subfolder basename")
    image_count: int = Field(..., description="Number of image files under this target")
    video_count: int = Field(..., description="Number of video files under this target")


class SplitPreviewResponse(BaseModel):
    """
    Preview of what a split would produce at a given depth.

    `blocked_reason` is populated when splitting is impossible (folder
    needs relink, active job, nothing to split, etc.). The frontend
    shows this as a disabled OK button with the reason displayed.
    """

    original_folder: str | None = Field(
        None, description="Absolute path of the deployment being split"
    )
    depth: int = Field(..., description="Chosen descent depth (1 = direct children)")
    max_depth: int = Field(
        ..., description="Deepest leaf depth reachable from the parent folder"
    )
    can_decrease: bool = Field(
        ..., description="True when depth > 1"
    )
    can_increase: bool = Field(
        ...,
        description="True when at least one branch has children below the current depth",
    )
    targets: list[SplitPreviewTarget] = Field(
        default_factory=list, description="Non-empty subfolders at the chosen depth"
    )
    blocked_reason: str | None = Field(
        None, description="Human-readable reason this deployment cannot be split"
    )


class SplitRequest(BaseModel):
    """Request body for POST /api/deployments/{id}/split."""

    depth: int = Field(..., ge=1, description="Descent depth (1 = direct children)")


class SplitResponse(BaseModel):
    """Result of a successful split."""

    created_deployment_ids: list[str] = Field(
        ..., description="IDs of the newly created child deployments"
    )
    message: str


class FolderPreviewResponse(BaseModel):
    """
    Preview of a deployment folder before running analysis.

    Provides quick counts, GPS location, and date range without storing files in DB.
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
    sample_files: list[SampleFile] = Field(
        [], description="Sample files with extracted datetimes for preview"
    )
    start_date: datetime | None = Field(
        None, description="Earliest DateTimeOriginal found in images"
    )
    end_date: datetime | None = Field(
        None, description="Latest DateTimeOriginal found in images"
    )
    missing_datetime: bool = Field(
        False, description="True if no EXIF datetime metadata was found"
    )
    datetime_validation_log: list[str] = Field(
        default_factory=list,
        description="Log of datetime extraction attempts and validation results",
    )
    mtime_start_date: datetime | None = Field(
        None,
        description=(
            "Earliest file modification time. Populated only when no capture "
            "dates were found, so the user can judge the opt-in fallback"
        ),
    )
    mtime_end_date: datetime | None = Field(
        None, description="Latest file modification time, same conditions"
    )
