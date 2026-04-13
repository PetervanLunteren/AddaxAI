"""Schemas for dashboard statistics endpoints."""

from datetime import date

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


class SpeciesObservationCount(BaseModel):
    """One species (or category) and its MaxN sum within a single deployment."""

    label: str
    label_taxonomy_id: str | None
    count: int


class ObservationRateMapFeature(BaseModel):
    """One deployment plotted on the map.

    `observation_count` is the sum of EventObservation.max_n across
    events that pass the active filters. `rate_per_100` is that count
    divided by trap nights * 100, matching the dashboard's metric.
    """

    deployment_id: str
    site_id: str
    site_name: str
    latitude: float
    longitude: float
    start_date: date
    end_date: date | None
    trap_nights: int
    observation_count: int
    rate_per_100: float
    species_breakdown: list[SpeciesObservationCount]


class ObservationRateMapResponse(BaseModel):
    features: list[ObservationRateMapFeature]
