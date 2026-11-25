"""Pydantic schemas for API request/response validation."""

from .detection import (
    DetectionCreate,
    DetectionResponse,
    DetectionResult,
    DetectionStatsResponse,
)
from .job import JobCreate, JobResponse, JobUpdate
from .project import ProjectCreate, ProjectResponse, ProjectUpdate
from .site import SiteCreate, SiteResponse, SiteUpdate

__all__ = [
    "DetectionCreate",
    "DetectionResponse",
    "DetectionResult",
    "DetectionStatsResponse",
    "JobCreate",
    "JobResponse",
    "JobUpdate",
    "ProjectCreate",
    "ProjectResponse",
    "ProjectUpdate",
    "SiteCreate",
    "SiteResponse",
    "SiteUpdate",
]
