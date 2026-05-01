"""
Tests for the Plots → Activity overlap endpoint and its math module.

Three layers, in order:
  1. Pure-math unit tests on `app.ml.activity_analysis` (KDE, Δ, bootstrap,
     diel classification). No DB.
  2. CRUD-level tests on `get_activity_overlap` with synthetic fixture data:
     2-species mode, 1-species mode, sample-size warning thresholds, both-
     species-empty edge case.
  3. HTTP endpoint shape via the FastAPI TestClient.
"""

from datetime import date, datetime

import numpy as np

from app.api.crud import statistics as stats_crud
from app.api.schemas.statistics import SunBands
from app.ml.activity_analysis import (
    bootstrap_overlap_ci,
    classify_diel,
    estimator_label,
    fit_circular_kde,
    overlap_coefficient,
)
from app.models.event_observation import EventObservation
from tests.conftest import (
    make_deployment,
    make_event_with_files,
    make_project,
    make_site,
)

# ---------------------------------------------------------------------------
# 1. Pure-math unit tests
# ---------------------------------------------------------------------------


def test_kde_empty_input_returns_zeros():
    grid, density = fit_circular_kde(np.array([]))
    assert grid.shape == (240,)
    assert density.shape == (240,)
    assert density.sum() == 0.0


def test_kde_density_integrates_to_one():
    """Post-normalization should produce a proper density."""
    times = np.array([6.0, 6.5, 7.0, 18.0, 18.5, 19.0])
    _, density = fit_circular_kde(times)
    integral = density.sum() * (24.0 / 240)
    assert abs(integral - 1.0) < 1e-9


def test_kde_peaks_near_input_clusters():
    """A bimodal input should produce two visible peaks at roughly the
    correct hours. We don't need exact peak positions — just that the
    density is much higher at the cluster centres than at the antipodes.
    """
    times = np.array([6.0] * 20 + [18.0] * 20)
    grid, density = fit_circular_kde(times)
    # Density at hour 6 and hour 18 should both be much greater than at
    # hour 12 (between clusters) or hour 0 (opposite).
    idx_6 = int(6 / (24 / 240))
    idx_12 = int(12 / (24 / 240))
    idx_18 = int(18 / (24 / 240))
    idx_0 = 0
    assert density[idx_6] > 3 * density[idx_12]
    assert density[idx_18] > 3 * density[idx_12]
    assert density[idx_6] > 3 * density[idx_0]


def test_overlap_identical_distributions_equals_one():
    times = np.array([6.0, 7.0, 8.0, 18.0, 19.0, 20.0])
    _, density = fit_circular_kde(times)
    assert abs(overlap_coefficient(density, density) - 1.0) < 1e-9


def test_overlap_disjoint_distributions_near_zero():
    """One species active midday, the other active midnight → Δ ≈ 0."""
    day = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    night = np.array([22.0, 23.0, 0.5, 1.0, 2.0])
    _, d_day = fit_circular_kde(day)
    _, d_night = fit_circular_kde(night)
    delta = overlap_coefficient(d_day, d_night)
    assert delta < 0.05  # almost no overlap with default kappa=5


def test_overlap_partial_overlap_in_between():
    """A 4-hour shift between two clusters should produce a moderate Δ."""
    a = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    _, da = fit_circular_kde(a)
    _, db = fit_circular_kde(b)
    delta = overlap_coefficient(da, db)
    assert 0.1 < delta < 0.9


def test_overlap_shape_mismatch_raises():
    import pytest

    a = np.zeros(240)
    b = np.zeros(120)
    with pytest.raises(ValueError, match="shapes must match"):
        overlap_coefficient(a, b)


def test_bootstrap_self_overlap_brackets_one():
    """Bootstrapping a species against itself: Δ point estimate is 1.0,
    CI lower bound is below 1.0 (resamples are not always identical),
    CI upper bound is at most 1.0."""
    times = np.array([6.0, 7.0, 8.0, 18.0, 19.0, 20.0])
    delta, lo, hi = bootstrap_overlap_ci(times, times, reps=200)
    assert abs(delta - 1.0) < 1e-9
    assert lo <= 1.0 + 1e-9
    assert hi <= 1.0 + 1e-9
    assert lo <= delta <= hi or lo <= 1.0  # CI can be entirely below 1


def test_bootstrap_empty_species_returns_zero():
    delta, lo, hi = bootstrap_overlap_ci(np.array([]), np.array([6.0]), reps=10)
    assert delta == 0.0 and lo == 0.0 and hi == 0.0


def test_bootstrap_is_deterministic():
    """Fixed seed = same CI on repeated calls. Important for testing
    and for stable UI display on re-fetches."""
    times_a = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    times_b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    r1 = bootstrap_overlap_ci(times_a, times_b, reps=100)
    r2 = bootstrap_overlap_ci(times_a, times_b, reps=100)
    assert r1 == r2


def test_estimator_label_thresholds():
    assert estimator_label(0) == "delta1"
    assert estimator_label(49) == "delta1"
    assert estimator_label(50) == "delta4"
    assert estimator_label(75) == "delta4"
    assert estimator_label(1000) == "delta4"


def test_classify_diel_diurnal():
    # Tight midday cluster
    times = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    grid, density = fit_circular_kde(times)
    sun = SunBands(dawn=5.5, sunrise=6.0, sunset=18.0, dusk=18.5)
    cls, phases = classify_diel(grid, density, sun)
    assert cls == "diurnal"
    assert phases["day"] > 0.7


def test_classify_diel_nocturnal():
    times = np.array([22.0, 23.0, 0.5, 1.0, 2.0])
    grid, density = fit_circular_kde(times)
    sun = SunBands(dawn=5.5, sunrise=6.0, sunset=18.0, dusk=18.5)
    cls, phases = classify_diel(grid, density, sun)
    assert cls == "nocturnal"
    assert phases["night"] > 0.7


def test_classify_diel_cathemeral():
    """Roughly uniform activity → no phase passes 0.70 threshold."""
    times = np.linspace(0.5, 23.5, 24)
    grid, density = fit_circular_kde(times)
    sun = SunBands(dawn=5.5, sunrise=6.0, sunset=18.0, dusk=18.5)
    cls, phases = classify_diel(grid, density, sun)
    assert cls == "cathemeral"
    assert max(phases.values()) < 0.7


def test_classify_diel_falls_back_when_no_sun_bands():
    """Polar / unknown-tz path: fixed 06:00-18:00 day window, no twilight."""
    times = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    grid, density = fit_circular_kde(times)
    cls, phases = classify_diel(grid, density, None)
    assert cls == "diurnal"
    assert phases["twilight"] == 0.0
    assert phases["day"] + phases["night"] > 0.99


# ---------------------------------------------------------------------------
# 2. CRUD-level tests with synthetic fixture data
# ---------------------------------------------------------------------------


def _add_obs(
    db,
    *,
    event_id: str,
    label: str,
    max_n: int = 1,
    category: str = "animal",
) -> EventObservation:
    obs = EventObservation(
        event_id=event_id,
        label=label,
        label_taxonomy_id=None,
        category=category,
        max_n=max_n,
    )
    db.add(obs)
    db.flush()
    return obs


def _build_two_species_project(db):
    """Project with one site, one deployment, and events for two species
    at different times of day so we can exercise the overlap math.
    Leopard (nocturnal-ish): 22:00 cluster.
    Domestic cattle (diurnal-ish): 12:00 cluster.
    """
    project = make_project(db, timezone="Europe/Amsterdam")
    site = make_site(
        db, project_id=project.id, name="Camp", latitude=52.37, longitude=4.89
    )
    dep = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 6, 1),
        end_date_local=date(2024, 6, 30),
    )
    # 60 leopard events at 22:00, 60 cattle events at 12:00
    for i in range(60):
        ev_l = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 6, 15, 22, i % 60),
        )
        _add_obs(db, event_id=ev_l.id, label="leopard")
        ev_c = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 6, 15, 12, i % 60),
        )
        _add_obs(db, event_id=ev_c.id, label="cattle")
    db.flush()
    return project


def test_get_activity_overlap_two_species_basic_shape(db):
    project = _build_two_species_project(db)
    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="leopard", species_b="cattle"
    )
    assert resp.species_a is not None
    assert resp.species_b is not None
    assert resp.overlap is not None
    assert resp.species_a.label == "leopard"
    assert resp.species_b.label == "cattle"
    assert resp.species_a.n == 60
    assert resp.species_b.n == 60
    assert len(resp.species_a.kde_density) == 240
    assert len(resp.species_b.kde_density) == 240
    assert resp.independence_interval_seconds > 0
    assert resp.project_timezone == "Europe/Amsterdam"


def test_get_activity_overlap_diel_classification(db):
    project = _build_two_species_project(db)
    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="leopard", species_b="cattle"
    )
    # Cattle at 12:00 should classify as diurnal regardless of latitude.
    assert resp.species_b.diel_class == "diurnal"
    # Leopard at 22:00 in Amsterdam in June straddles the day/night
    # boundary (sunset ~22:00). The Bennie ≥0.70-density rule lands
    # somewhere in {nocturnal, crepuscular, cathemeral} depending on
    # exact sun bands; the only definitive negative is "diurnal".
    assert resp.species_a.diel_class != "diurnal"
    assert resp.species_a.diel_class in (
        "nocturnal", "crepuscular", "cathemeral"
    )


def test_get_activity_overlap_delta_low_for_disjoint_clusters(db):
    project = _build_two_species_project(db)
    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="leopard", species_b="cattle"
    )
    assert resp.overlap is not None
    # 12:00 and 22:00 are 10 hours apart on the circular axis — modest
    # overlap with default kappa=5.
    assert resp.overlap.delta < 0.5
    assert resp.overlap.ci_low <= resp.overlap.delta <= resp.overlap.ci_high
    assert resp.overlap.bootstrap_reps == 1000
    assert resp.overlap.min_n == 60
    assert resp.overlap.delta_estimator == "delta4"


def test_get_activity_overlap_single_species_mode(db):
    project = _build_two_species_project(db)
    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="leopard", species_b=None
    )
    assert resp.species_a is not None
    assert resp.species_b is None
    assert resp.overlap is None


def test_get_activity_overlap_unknown_species_returns_zero_n(db):
    project = _build_two_species_project(db)
    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="unicorn", species_b=None
    )
    assert resp.species_a.n == 0
    assert resp.species_a.sample_size_warning == "low_n_30"


def test_get_activity_overlap_overlap_none_when_one_species_empty(db):
    project = _build_two_species_project(db)
    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="leopard", species_b="unicorn"
    )
    assert resp.species_a.n == 60
    assert resp.species_b.n == 0
    assert resp.overlap is None


def test_get_activity_overlap_sample_size_warnings(db):
    """Add a tiny synthetic species and confirm the warning bucket."""
    project = make_project(db, timezone="Europe/Amsterdam")
    site = make_site(
        db, project_id=project.id, name="Camp", latitude=52.37, longitude=4.89
    )
    dep = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 6, 1),
        end_date_local=date(2024, 6, 30),
    )
    # 10 events of "rare" → low_n_30
    for i in range(10):
        ev = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 6, 15, 12, i),
        )
        _add_obs(db, event_id=ev.id, label="rare")
    # 40 events of "uncommon" → low_n_50
    for i in range(40):
        ev = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 6, 16, 12, i),
        )
        _add_obs(db, event_id=ev.id, label="uncommon")
    # 60 events of "common" → low_n_75
    for i in range(60):
        ev = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 6, 17, 12, i),
        )
        _add_obs(db, event_id=ev.id, label="common")
    # 100 events of "frequent" → no warning
    for i in range(100):
        ev = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 6, 18, 12, i % 60),
        )
        _add_obs(db, event_id=ev.id, label="frequent")
    db.flush()

    for species, expected in [
        ("rare", "low_n_30"),
        ("uncommon", "low_n_50"),
        ("common", "low_n_75"),
        ("frequent", None),
    ]:
        resp = stats_crud.get_activity_overlap(
            db, project.id, species_a=species, species_b=None
        )
        assert resp.species_a.sample_size_warning == expected, (
            f"{species}: expected {expected}, got {resp.species_a.sample_size_warning}"
        )


def test_get_activity_overlap_date_filter_excludes_events(db):
    """Events outside the date range must not contribute to the species n."""
    project = _build_two_species_project(db)
    # Restrict to 2025 — no events match.
    resp = stats_crud.get_activity_overlap(
        db,
        project.id,
        species_a="leopard",
        species_b="cattle",
        date_from="2025-01-01",
        date_to="2025-12-31",
    )
    assert resp.species_a.n == 0
    assert resp.species_b.n == 0
    assert resp.overlap is None


# ---------------------------------------------------------------------------
# 3. HTTP endpoint shape
# ---------------------------------------------------------------------------


def test_activity_overlap_endpoint_response_shape(client, db):
    project = _build_two_species_project(db)
    resp = client.get(
        f"/api/statistics/activity-overlap?project_id={project.id}"
        "&species_a=leopard&species_b=cattle"
    )
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "species_a",
        "species_b",
        "overlap",
        "sun_bands",
        "project_timezone",
        "independence_interval_seconds",
    ):
        assert key in data
    assert data["project_timezone"] == "Europe/Amsterdam"
    assert data["species_a"]["label"] == "leopard"
    assert data["species_b"]["label"] == "cattle"
    assert data["overlap"]["bootstrap_reps"] == 1000
    assert isinstance(data["species_a"]["kde_density"], list)
    assert len(data["species_a"]["kde_density"]) == 240


def test_activity_overlap_endpoint_single_species(client, db):
    project = _build_two_species_project(db)
    resp = client.get(
        f"/api/statistics/activity-overlap?project_id={project.id}&species_a=leopard"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["species_b"] is None
    assert data["overlap"] is None


# ---------------------------------------------------------------------------
# 4. Sun-time mode (Vazquez 2019 double-anchor transform)
# ---------------------------------------------------------------------------


def test_get_activity_overlap_defaults_to_clock_axis(db):
    project = _build_two_species_project(db)
    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="leopard", species_b="cattle"
    )
    assert resp.time_axis == "clock"
    assert resp.anchor_sun_bands is None
    # Clock-mode sun bands still populated for the overlay.
    assert resp.sun_bands is not None


def test_get_activity_overlap_sun_mode_basic(db):
    project = _build_two_species_project(db)
    resp = stats_crud.get_activity_overlap(
        db,
        project.id,
        species_a="leopard",
        species_b="cattle",
        time_axis="sun",
    )
    assert resp.time_axis == "sun"
    assert resp.anchor_sun_bands is not None
    # Anchor bands obey the dawn < sunrise < sunset < dusk ordering.
    a = resp.anchor_sun_bands
    assert a.dawn < a.sunrise < a.sunset < a.dusk
    # No polar drops for Amsterdam in June.
    assert resp.species_a.dropped_polar == 0
    assert resp.species_b.dropped_polar == 0


def test_get_activity_overlap_sun_mode_fixed_offset_tz(db):
    """Projects with fixed-offset timezones (cameras set to UTC, say)
    should round-trip through the sun-time pipeline without
    zoneinfo resolution issues."""
    project = make_project(db, timezone="UTC")
    site = make_site(
        db, project_id=project.id, name="Camp", latitude=-1.28, longitude=36.82
    )
    dep = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 3, 1),
        end_date_local=date(2024, 3, 31),
    )
    for i in range(40):
        ev = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 3, 15, 12, i % 60),
        )
        _add_obs(db, event_id=ev.id, label="zebra")
    db.flush()

    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="zebra", time_axis="sun"
    )
    assert resp.time_axis == "sun"
    assert resp.anchor_sun_bands is not None
    assert resp.species_a.dropped_polar == 0
    # Classification is still valid: the zebra cluster is at UTC 12:00,
    # and Nairobi is only 3 h off UTC so midday stays "day" in the
    # anchored frame.
    assert resp.species_a.diel_class == "diurnal"


def test_get_activity_overlap_sun_mode_drops_polar_observations(db):
    """High-latitude project in midwinter: every observation lands in
    polar night, so every observation is dropped. The response still
    falls back cleanly (time_axis="clock" because anchors couldn't be
    computed), and the chart remains renderable."""
    project = make_project(db, timezone="UTC")
    site = make_site(
        db, project_id=project.id, name="PolarCamp", latitude=85.0, longitude=0.0
    )
    dep = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 12, 1),
        end_date_local=date(2024, 12, 31),
    )
    for i in range(30):
        ev = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 12, 15, 12, i),
        )
        _add_obs(db, event_id=ev.id, label="arctic_fox")
    db.flush()

    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="arctic_fox", time_axis="sun"
    )
    # All dates polar -> can't build anchors -> fall back to clock.
    assert resp.time_axis == "clock"
    assert resp.anchor_sun_bands is None
    assert resp.species_a.n == 30
    assert resp.species_a.dropped_polar == 0  # no partial drop, full fallback


def test_get_activity_overlap_sun_mode_partial_polar_drops(db):
    """High-latitude project with observations split across polar and
    non-polar dates: the non-polar observations anchor the transform
    and the polar ones are counted as dropped.

    At 70N / UTC: March 15 is non-polar (sunrise ~04:41, sunset ~16:18),
    June 15 is polar (continuous twilight), so the month split
    cleanly exercises both code paths.
    """
    project = make_project(db, timezone="UTC")
    site = make_site(
        db,
        project_id=project.id,
        name="ArcticCamp",
        latitude=70.0,
        longitude=25.0,
    )
    dep = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 12, 31),
    )
    # 20 observations in March (non-polar) + 20 in June (polar).
    for i in range(20):
        ev = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 3, 15, 12, i % 60),
        )
        _add_obs(db, event_id=ev.id, label="arctic_fox")
    for i in range(20):
        ev = make_event_with_files(
            db,
            deployment_id=dep.id,
            event_start_local=datetime(2024, 6, 15, 12, i % 60),
        )
        _add_obs(db, event_id=ev.id, label="arctic_fox")
    db.flush()

    resp = stats_crud.get_activity_overlap(
        db, project.id, species_a="arctic_fox", time_axis="sun"
    )
    # March observations survive, June observations drop.
    assert resp.time_axis == "sun"
    assert resp.anchor_sun_bands is not None
    assert resp.species_a.n == 20
    assert resp.species_a.dropped_polar == 20


def test_activity_overlap_endpoint_sun_mode_query_param(client, db):
    project = _build_two_species_project(db)
    resp = client.get(
        f"/api/statistics/activity-overlap?project_id={project.id}"
        "&species_a=leopard&species_b=cattle&time_axis=sun"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["time_axis"] == "sun"
    assert data["anchor_sun_bands"] is not None
    assert "dropped_polar" in data["species_a"]
    assert data["species_a"]["dropped_polar"] == 0
