"""
Event schemas for API requests and responses.
"""

from datetime import datetime

from pydantic import BaseModel

from app.api.schemas.file import FileWithDetections


class EventSummary(BaseModel):
    """Event summary for browse card display."""

    id: str
    deployment_id: str
    start_time: datetime
    end_time: datetime
    file_count: int
    representative_file_id: str | None
    species: list[str]
    observation_type: str
    verified_count: int
    total_count: int


class EventWithFiles(BaseModel):
    """Event with all files and their detections."""

    id: str
    deployment_id: str
    start_time: datetime
    end_time: datetime
    file_count: int
    representative_file_id: str | None = None
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


class EventFilterOptions(BaseModel):
    """Available filter options for a project's events."""

    species: list[str]
    date_range: DateRange | None
