"""
Event schemas for API requests and responses.
"""

from datetime import datetime

from pydantic import BaseModel

from app.api.schemas.file import FileWithDetections


class MaxNFrame(BaseModel):
    """A MaxN frame reference for filmstrip badges."""

    file_id: str
    label: str
    max_n: int


class EventSummary(BaseModel):
    """Event summary for browse card display."""

    id: str
    deployment_id: str
    start_time: datetime
    end_time: datetime
    file_count: int
    thumbnail_file_id: str | None
    max_n_frames: list[MaxNFrame]
    site_name: str | None
    labels: list[str]
    display_labels: dict[str, str] | None = None
    observation_type: str
    observation_types: list[str]
    image_count: int
    frame_count: int
    video_count: int
    verified_count: int
    total_count: int
    verified_maxn_count: int
    total_maxn_count: int


class EventWithFiles(BaseModel):
    """Event with all files and their detections."""

    id: str
    deployment_id: str
    start_time: datetime
    end_time: datetime
    file_count: int
    max_n_frames: list[MaxNFrame]
    created_at: datetime
    site_name: str | None = None
    files: list[FileWithDetections]

    class Config:
        from_attributes = True


class GenerateEventsRequest(BaseModel):
    """Request body for event generation."""

    project_id: str


class GenerateEventsResponse(BaseModel):
    """Response for event generation."""

    event_count: int
    message: str


class AdjacentEventsResponse(BaseModel):
    """Response for adjacent event navigation."""

    previous_id: str | None
    next_id: str | None
    next_unverified_id: str | None
    current_index: int
    total_count: int


class DateRange(BaseModel):
    """Date range with min and max."""

    min: datetime
    max: datetime


class EventVerificationStats(BaseModel):
    """Aggregate verification stats across filtered events."""

    total_files: int
    verified_files: int
    total_max_n_frames: int
    verified_max_n_frames: int
    total_observations: int
    verified_detections: int


class EventFilterOptions(BaseModel):
    """Available filter options for a project's events."""

    labels: list[str]
    date_range: DateRange | None
    label_event_counts: dict[str, int]


class LabelTreeNode(BaseModel):
    """A node in the label filter tree."""

    id: str
    name: str
    level: int
    children: list["LabelTreeNode"]
    selected: bool
    annotation: str | None = None
    count: int | None = None
    child_count: int | None = None


class LabelTreeResponse(BaseModel):
    """Response for the label filter tree endpoint."""

    tree: list[LabelTreeNode]
    all_leaf_ids: list[str]
    label_event_counts: dict[str, int]
    count_unit: str
