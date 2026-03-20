"""Schemas for dashboard statistics endpoints."""

from pydantic import BaseModel


class DashboardOverview(BaseModel):
    total_files: int
    total_observations: int
    total_events: int
    total_deployments: int
    total_sites: int
    trap_nights: int
    first_file_date: str | None
    last_file_date: str | None


class SpeciesCount(BaseModel):
    species: str
    count: int


class HourlyCount(BaseModel):
    hour: int
    count: int


class ActivityPatternResponse(BaseModel):
    hours: list[HourlyCount]
    total_observations: int


class DetectionTrendPoint(BaseModel):
    date: str
    count: int


class DetectionCategories(BaseModel):
    animal_count: int
    person_count: int
    vehicle_count: int
    empty_count: int


class VerificationProgress(BaseModel):
    total_files: int
    verified_files: int
