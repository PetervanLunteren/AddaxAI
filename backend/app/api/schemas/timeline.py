"""Schemas for the deployment-timeline insight endpoint."""

from datetime import date

from pydantic import BaseModel


class TrapNightInterval(BaseModel):
    start: date
    end: date
    trap_nights: int


class TimelineDeployment(BaseModel):
    deployment_id: str
    deployment_label: str
    camera_model: str | None
    configured_start: date
    configured_end: date | None
    intervals: list[TrapNightInterval]
    file_count: int


class TimelineSite(BaseModel):
    site_id: str | None
    site_name: str
    deployments: list[TimelineDeployment]


class ConcurrentPoint(BaseModel):
    date: date
    count: int


class HeatmapPoint(BaseModel):
    """One site's media-file count for one calendar day.

    Only days with at least one file get a point, so the payload is bounded
    by (sites x days with files). `site_id` is None for the "(no site)" row,
    matching `TimelineSite.site_id`.
    """

    site_id: str | None
    date: date
    count: int


class TimelineMetrics(BaseModel):
    site_count: int
    deployment_count: int
    total_trap_nights: int
    median_deployment_length_days: float | None
    max_concurrent_cameras: int


class TimelineResponse(BaseModel):
    sites: list[TimelineSite]
    concurrent_cameras: list[ConcurrentPoint]
    heatmap: list[HeatmapPoint]
    metrics: TimelineMetrics
    date_range_from: date | None
    date_range_to: date | None
