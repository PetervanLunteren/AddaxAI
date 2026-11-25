"""
Job schemas for API requests and responses.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit validation
- Clear error messages
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Job types
JobType = Literal[
    "deployment_analysis",
    "import",
    "ml_inference",
    "export",
    "event_computation",
    "postprocessing",
    "re_embedding",
    "camtrap_export",
    "recognition_json_export",
    "folder_run_save_outputs",
    "media_export",
    "package_export",
    "package_import",
]
JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class JobCreate(BaseModel):
    """Schema for creating a new job."""

    type: JobType = Field(..., description="Type of job")
    payload: dict = Field(..., description="Job-specific parameters")


class JobUpdate(BaseModel):
    """Schema for updating a job."""

    status: JobStatus | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    result: dict | None = None
    error: str | None = None


class JobResponse(BaseModel):
    """Schema for job response."""

    id: str
    type: str
    status: str
    progress_current: int
    progress_total: int | None
    payload: dict | None
    result: dict | None
    error: str | None
    created_at_utc: datetime
    started_at_utc: datetime | None
    completed_at_utc: datetime | None

    class Config:
        from_attributes = True
