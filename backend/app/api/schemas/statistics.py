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


class SunBands(BaseModel):
    """Fractional-hour timestamps for civil twilight + day boundaries.

    Computed server-side via python-astral from the project's averaged
    site lat/lon, its IANA timezone, and a reference date drawn from
    the filter range midpoint. Fed to the Activity pattern chart so
    the frontend can color each hour bar as night / dawn-dusk / day.
    """

    dawn: float
    sunrise: float
    sunset: float
    dusk: float


class ActivityPatternResponse(BaseModel):
    hours: list[HourlyCount]
    total_observations: int
    sun_bands: SunBands | None = None


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
    start_date_local: date
    end_date_local: date | None
    trap_nights: int
    observation_count: int
    rate_per_100: float
    species_breakdown: list[SpeciesObservationCount]


class ObservationRateMapResponse(BaseModel):
    features: list[ObservationRateMapFeature]
