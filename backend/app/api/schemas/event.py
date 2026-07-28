"""
Event schemas for API requests and responses.
"""

from datetime import datetime

from pydantic import BaseModel, field_serializer

from app.api.schemas.file import FileWithDetections
from app.utils.datetime_serialization import serialize_local_datetime


class MaxNFrame(BaseModel):
    """A MaxN frame reference for filmstrip badges."""

    file_id: str
    label: str | None = None
    label_taxonomy_id: str | None = None
    max_n: int
    # Human-authoritative count (`human_count` if set, else `max_n`).
    effective_count: int


class EventSummary(BaseModel):
    """Event summary for browse card display."""

    id: str
    deployment_id: str
    # Null when the event's files have no extractable capture time. Missing
    # capture time is tolerated, not fatal (see DEVELOPERS.md datetime conventions);
    # these events drop out of time-based features but must still serialize.
    event_start_local: datetime | None
    event_end_local: datetime | None
    file_count: int
    thumbnail_file_id: str | None
    # Up to four file IDs used for the event-card collage. First slots
    # come from `max_n_frames` (one per dominant species), remaining
    # slots are padded by max detection confidence. Empty for events
    # with no files.
    collage_file_ids: list[str] = []
    max_n_frames: list[MaxNFrame]
    site_name: str | None
    labels: list[str]
    scientific_labels: dict[str, str] | None = None
    common_labels: dict[str, str] | None = None
    observation_type: str
    observation_types: list[str]
    image_count: int
    frame_count: int
    video_count: int
    verified_count: int
    total_count: int
    verified_maxn_count: int
    total_maxn_count: int
    # Human confirmation of the event's species and counts, stored on
    # Event.confirmed (the "Confirm" action on the Observations page). The
    # verified_count / verified_maxn_count fields above are file-level
    # secondary detail.
    is_confirmed: bool
    # Aggregated file-level state for the card corner cluster.
    any_file_flagged: bool
    any_file_favorited: bool

    # event_start_local / event_end_local are naive wall-clock times in
    # the project's local camera timezone (see DEVELOPERS.md).
    @field_serializer("event_start_local", "event_end_local")
    def _serialize_event_local(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)


class EventObservationItem(BaseModel):
    """One species row of an event's count list (the Observations-page
    count editor). `effective_count` is the human-authoritative count
    (`human_count` if set, else the AI-derived `max_n`). A human-only
    species the AI missed has max_n=0."""

    id: str
    category: str
    label: str | None = None
    label_taxonomy_id: str | None = None
    common_name: str | None = None
    scientific_name: str | None = None
    max_n: int
    effective_count: int


class EventWithFiles(BaseModel):
    """Event with all files and their detections."""

    id: str
    deployment_id: str
    # Null when the event's files have no extractable capture time. Missing
    # capture time is tolerated, not fatal (see DEVELOPERS.md datetime conventions);
    # these events drop out of time-based features but must still serialize.
    event_start_local: datetime | None
    event_end_local: datetime | None
    file_count: int
    max_n_frames: list[MaxNFrame]
    # Human confirmation of the species and counts ("Confirm" action).
    confirmed: bool = False
    # Per-species count list (AI + human-only), highest count first.
    observations: list[EventObservationItem] = []
    created_at_utc: datetime
    site_name: str | None = None
    files: list[FileWithDetections]

    @field_serializer("event_start_local", "event_end_local")
    def _serialize_event_local(self, value: datetime | None) -> str | None:
        return serialize_local_datetime(value)

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
    next_unconfirmed_id: str | None
    current_index: int
    total_count: int


class DateRange(BaseModel):
    """Date range with min and max."""

    min: datetime
    max: datetime


class EventVerificationStats(BaseModel):
    """Aggregate verification stats across filtered events.

    The Events tab progress bar reads `events_confirmed` /
    `events_total`. The other fields remain for any downstream
    consumer that needs file-level granularity.
    """

    events_confirmed: int
    events_total: int
    total_files: int
    verified_files: int
    total_max_n_frames: int
    verified_max_n_frames: int
    total_observations: int
    total_detections: int
    verified_detections: int


class EventFilterOptions(BaseModel):
    """Available filter options for a project's events."""

    labels: list[str]
    date_range: DateRange | None
    label_event_counts: dict[str, int]
    scientific_labels: dict[str, str] | None = None
    common_labels: dict[str, str] | None = None
    # Lowest classification confidence in the project; the filter bars'
    # data-driven clamp for the classification range slider. None when
    # no classifications exist yet.
    min_label_confidence: float | None = None


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
