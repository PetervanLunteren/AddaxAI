"""Tests for the activity-pattern endpoint with sun band overlay.

Covers:
- Sun bands returned when the project has sites with GPS + a
  valid IANA timezone.
- None fallback when the project has no sites.
- Reference date respects the filter range midpoint (December
  band differs from June band at mid-latitudes).
- Existing hourly count behaviour is unchanged.
"""

from datetime import date, datetime

from app.api.crud import statistics as stats_crud
from app.models.event_observation import EventObservation
from tests.conftest import (
    make_deployment,
    make_event_with_files,
    make_project,
    make_site,
)


def _add_observation(
    db, *, event_id: str, label: str = "leopard", max_n: int = 1
) -> EventObservation:
    obs = EventObservation(
        event_id=event_id,
        label=label,
        label_taxonomy_id=None,
        category="animal",
        max_n=max_n,
    )
    db.add(obs)
    db.flush()
    return obs


def _build_netherlands_project(db):
    """Project with one site at Amsterdam, timezone Europe/Amsterdam."""
    project = make_project(db, timezone="Europe/Amsterdam")
    site = make_site(
        db, project_id=project.id, name="Amsterdam", latitude=52.37, longitude=4.89
    )
    dep = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 12, 31),
    )
    # One event so total_observations > 0
    ev = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 6, 15, 7, 30)
    )
    _add_observation(db, event_id=ev.id, max_n=3)
    db.flush()
    return project


def test_activity_pattern_returns_sun_bands(db):
    project = _build_netherlands_project(db)
    response = stats_crud.get_activity_pattern(db, project.id)

    assert response.sun_bands is not None
    bands = response.sun_bands
    assert 0 <= bands.dawn < bands.sunrise < bands.sunset < bands.dusk < 24


def test_activity_pattern_sun_bands_reflect_season(db):
    """At 52°N, summer sunrise is much earlier than winter sunrise."""
    project = _build_netherlands_project(db)

    summer = stats_crud.get_activity_pattern(
        db, project.id, date_from="2024-06-01", date_to="2024-06-30"
    )
    winter = stats_crud.get_activity_pattern(
        db, project.id, date_from="2024-12-01", date_to="2024-12-31"
    )

    assert summer.sun_bands is not None
    assert winter.sun_bands is not None

    # Summer sunrise should be at least 3 hours earlier than winter sunrise.
    # Amsterdam actual values: ~5:20 in late June vs ~8:45 in late December.
    assert summer.sun_bands.sunrise < winter.sun_bands.sunrise - 3

    # Summer sunset should be much later than winter sunset.
    assert summer.sun_bands.sunset > winter.sun_bands.sunset + 3


def test_activity_pattern_sun_bands_null_when_no_sites(db):
    """Project without sites produces None sun_bands without crashing."""
    project = make_project(db, timezone="Europe/Amsterdam")
    response = stats_crud.get_activity_pattern(db, project.id)
    assert response.sun_bands is None
    assert response.total_observations == 0


def test_activity_pattern_hourly_counts_unchanged(db):
    """Adding sun_bands doesn't break the existing hourly count payload."""
    project = _build_netherlands_project(db)
    response = stats_crud.get_activity_pattern(db, project.id)

    assert len(response.hours) == 24
    assert [h.hour for h in response.hours] == list(range(24))
    # The one event at 07:30 puts max_n=3 into hour 7
    assert response.hours[7].count == 3
    assert response.total_observations == 3


def test_activity_pattern_endpoint_response_shape(client, db):
    project = _build_netherlands_project(db)
    resp = client.get(
        f"/api/statistics/activity-pattern?project_id={project.id}"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "hours" in data
    assert "total_observations" in data
    assert "sun_bands" in data
    assert data["sun_bands"] is not None
    for key in ("dawn", "sunrise", "sunset", "dusk"):
        assert key in data["sun_bands"]
        assert isinstance(data["sun_bands"][key], float)
