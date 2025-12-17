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
JobType = Literal["deployment_analysis", "import", "ml_inference", "export", "event_computation"]
JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# Model options for deployment analysis
DetectionModel = Literal["MegaDetector 5A", "MegaDetector v1000 Redwood"]
ClassificationModel = Literal["Europe", "Africa"]


class DeploymentAnalysisPayload(BaseModel):
    """Payload for deployment_analysis job type."""

    project_id: str = Field(..., description="ID of the project (site will be auto-created)")
    folder_path: str = Field(..., description="Absolute path to deployment folder")
    detection_model: DetectionModel = Field(..., description="Detection model to use")
    classification_model: ClassificationModel = Field(..., description="Classification model to use")


class JobCreate(BaseModel):
    """Schema for creating a new job."""

    type: JobType = Field(..., description="Type of job")
    payload: dict = Field(..., description="Job-specific parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "deployment_analysis",
                "payload": {
                    "project_id": "abc-123",
                    "folder_path": "/Users/peter/camera-traps/site-a",
                    "detection_model": "MegaDetector 5A",
                    "classification_model": "Europe",
                },
            }
        }


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
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    class Config:
        from_attributes = True


class RunQueueResponse(BaseModel):
    """Response for running the job queue."""

    message: str = Field(..., description="Status message")
    jobs_started: int = Field(..., description="Number of jobs started")
    job_ids: list[str] = Field(..., description="IDs of jobs that were started")
