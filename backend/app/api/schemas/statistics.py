"""Schemas for dashboard statistics endpoints."""

from datetime import date
from typing import Literal

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
    # Number of deployments in the filtered set that have no site
    # assigned. They are silently excluded from the sun-time
    # computation; the UI renders a banner when this is non-zero.
    deployments_without_site: int = 0


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
    Features are only built for deployments that have a site, so
    `site_id`, `site_name`, `latitude` and `longitude` are non-null.
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
    # Number of deployments that otherwise passed the filters but had
    # no camera site assigned. Surfaced so the UI can render a
    # "X deployments without a site" banner.
    deployments_without_site: int = 0


# ---------------------------------------------------------------------------
# Activity overlap (Plots → Activity overlap page)
# ---------------------------------------------------------------------------


SampleSizeWarning = Literal["low_n_30", "low_n_50", "low_n_75"]
DielClass = Literal["diurnal", "nocturnal", "crepuscular", "cathemeral"]
DeltaEstimator = Literal["delta1", "delta4"]
TimeAxis = Literal["clock", "sun"]


class SpeciesActivity(BaseModel):
    """Per-species inputs to the activity-overlap chart.

    `kde_density` is the von Mises KDE evaluated on a 240-point grid
    over [0, 24) hours (matching `ml.activity_analysis.KDE_GRID_SAMPLES`),
    normalized so the integral over the grid is 1.0. The frontend draws
    this as a smooth curve.

    `raw_detection_times` is the underlying sample (decimal hours) used
    for the rug ticks under the curve. Capped at 5000 entries to bound
    payload size on huge datasets — the rug only needs visual density,
    not every single tick.

    `diel_density_by_phase` keys are "day", "night", "twilight" and sum
    to ~1.0. Surfaced so a UI tooltip can show the exact rule that
    produced the diel classification.
    """

    label: str
    n: int
    raw_detection_times: list[float]
    kde_density: list[float]
    diel_class: DielClass
    diel_density_by_phase: dict[str, float]
    sample_size_warning: SampleSizeWarning | None = None
    # Count of observations skipped because their date had no defined
    # sunrise (polar night / day). Always 0 in clock mode.
    dropped_polar: int = 0


class OverlapStat(BaseModel):
    """Pairwise activity overlap coefficient and bootstrap CI.

    Δ = ∫ min(f_a, f_b) dt over [0, 24], where f_a and f_b are the
    species' KDE densities. CI is from a 1000-rep percentile bootstrap.
    `delta_estimator` is the conventional Ridout & Linkie 2009 label
    (delta4 above min-N=50, delta1 below) — the underlying KDE method
    is the same for both, the label tells the reader which name to cite.
    """

    delta_estimator: DeltaEstimator
    delta: float
    ci_low: float
    ci_high: float
    bootstrap_reps: int
    min_n: int


class ActivityOverlapResponse(BaseModel):
    """Full payload for the Plots → Activity overlap page.

    `species_b` is None when the user has only picked one species; the
    chart still renders one curve in that case. `overlap` is None
    whenever we don't have two non-empty species to compare.

    `project_timezone` and `independence_interval_seconds` are
    read-only echoes of the two project settings that the user needs
    to know when reading the chart (x-axis interpretation and event
    grouping). The frontend surfaces them in the footer.
    """

    species_a: SpeciesActivity
    species_b: SpeciesActivity | None
    overlap: OverlapStat | None
    sun_bands: SunBands | None
    # Reference date the clock-mode `sun_bands` were computed for
    # (the midpoint of the filter range by default). Echoed so the
    # chart can caption the bands with the day they represent. Null
    # when sun_bands is null.
    sun_bands_reference_date: date | None = None
    # Mean dawn / sunrise / sunset / dusk across every observation's
    # date. Populated only in sun-time mode; the chart uses it to paint
    # twilight bands at the anchor positions.
    anchor_sun_bands: SunBands | None = None
    # Axis convention of the returned KDE: "clock" = raw wall-clock
    # hour, "sun" = Vazquez-anchored sun time.
    time_axis: TimeAxis = "clock"
    project_timezone: str
    independence_interval_seconds: int
    # Number of deployments in the filtered set that have no site
    # assigned. They are silently excluded from sun-time averaging;
    # the UI renders a banner when this is non-zero and sun mode is
    # active.
    deployments_without_site: int = 0
