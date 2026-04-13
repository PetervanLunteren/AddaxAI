"""Tests for the observation-rate-map endpoint and its CRUD function.

Builds a tiny fixture project with two deployments at different sites,
exercises the filtering options, and asserts the rate computation
matches the dashboard's MaxN-per-event metric.
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
    db,
    *,
    event_id: str,
    label: str,
    max_n: int,
    label_taxonomy_id: str | None = None,
    category: str = "animal",
) -> EventObservation:
    obs = EventObservation(
        event_id=event_id,
        label=label,
        label_taxonomy_id=label_taxonomy_id,
        category=category,
        max_n=max_n,
    )
    db.add(obs)
    db.flush()
    return obs


def _build_fixture(db):
    """One project, two sites, two deployments, a few events with MaxN."""
    project = make_project(db)

    site_a = make_site(
        db, project_id=project.id, name="Alpha", latitude=10.0, longitude=20.0
    )
    site_b = make_site(
        db, project_id=project.id, name="Beta", latitude=11.0, longitude=21.0
    )

    dep_a = make_deployment(
        db,
        site_id=site_a.id,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 11),  # 10 nights
    )
    dep_b = make_deployment(
        db,
        site_id=site_b.id,
        start_date=date(2024, 2, 1),
        end_date=date(2024, 2, 6),  # 5 nights
    )

    # Two events at deployment A
    ev_a1 = make_event_with_files(
        db, deployment_id=dep_a.id, start_time=datetime(2024, 1, 2, 8, 0)
    )
    ev_a2 = make_event_with_files(
        db, deployment_id=dep_a.id, start_time=datetime(2024, 1, 5, 14, 0)
    )

    # One event at deployment B
    ev_b1 = make_event_with_files(
        db, deployment_id=dep_b.id, start_time=datetime(2024, 2, 3, 9, 0)
    )

    # MaxN observations
    _add_observation(db, event_id=ev_a1.id, label="leopard", max_n=2)
    _add_observation(db, event_id=ev_a2.id, label="leopard", max_n=3)
    _add_observation(db, event_id=ev_b1.id, label="lion", max_n=1)

    db.flush()
    return project, site_a, site_b, dep_a, dep_b


# ---------------------------------------------------------------------------
# get_per_deployment_trap_nights
# ---------------------------------------------------------------------------


def test_per_deployment_trap_nights_sums_to_total(db):
    project, _, _, dep_a, dep_b = _build_fixture(db)

    per_dep = stats_crud.get_per_deployment_trap_nights(db, project.id)
    assert per_dep[dep_a.id] == 10
    assert per_dep[dep_b.id] == 5

    total = stats_crud.get_trap_nights(db, project.id)
    assert total == 15


# ---------------------------------------------------------------------------
# get_observation_rate_map
# ---------------------------------------------------------------------------


def test_observation_rate_map_returns_one_feature_per_deployment(db):
    project, site_a, site_b, dep_a, dep_b = _build_fixture(db)

    response = stats_crud.get_observation_rate_map(db, project.id)
    assert len(response.features) == 2

    by_id = {f.deployment_id: f for f in response.features}
    feature_a = by_id[dep_a.id]
    assert feature_a.site_name == "Alpha"
    assert feature_a.latitude == 10.0
    assert feature_a.longitude == 20.0
    assert feature_a.trap_nights == 10
    assert feature_a.observation_count == 5  # 2 + 3
    assert feature_a.rate_per_100 == 50.0  # 5 / 10 * 100

    feature_b = by_id[dep_b.id]
    assert feature_b.site_name == "Beta"
    assert feature_b.trap_nights == 5
    assert feature_b.observation_count == 1
    assert feature_b.rate_per_100 == 20.0  # 1 / 5 * 100


def test_observation_rate_map_filters_by_site(db):
    project, site_a, _, dep_a, _ = _build_fixture(db)

    response = stats_crud.get_observation_rate_map(
        db, project.id, site_ids=[site_a.id]
    )
    assert len(response.features) == 1
    assert response.features[0].deployment_id == dep_a.id


def test_observation_rate_map_filters_by_date(db):
    project, _, _, dep_a, dep_b = _build_fixture(db)

    # Date range that only catches deployment A's events.
    response = stats_crud.get_observation_rate_map(
        db, project.id, date_from="2024-01-01", date_to="2024-01-31"
    )
    by_id = {f.deployment_id: f for f in response.features}

    # Deployment A: events in range, rate computed on clipped trap nights.
    assert dep_a.id in by_id
    assert by_id[dep_a.id].observation_count == 5

    # Deployment B: no events in range, but it does have effort... actually
    # its dates fall after 2024-01-31, so its trap nights are clipped to 0.
    # Should be dropped.
    assert dep_b.id not in by_id


def test_observation_rate_map_includes_species_breakdown(db):
    project, _, _, dep_a, dep_b = _build_fixture(db)

    response = stats_crud.get_observation_rate_map(db, project.id)
    by_id = {f.deployment_id: f for f in response.features}

    leopard = by_id[dep_a.id].species_breakdown
    assert len(leopard) == 1
    assert leopard[0].label == "leopard"
    assert leopard[0].count == 5

    lion = by_id[dep_b.id].species_breakdown
    assert len(lion) == 1
    assert lion[0].label == "lion"
    assert lion[0].count == 1


def test_observation_rate_map_endpoint(client, db):
    project, _, _, _, _ = _build_fixture(db)

    resp = client.get(
        "/api/statistics/observation-rate-map",
        params={"project_id": project.id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "features" in data
    assert len(data["features"]) == 2
    feature = data["features"][0]
    for key in (
        "deployment_id",
        "site_id",
        "site_name",
        "latitude",
        "longitude",
        "start_date",
        "end_date",
        "trap_nights",
        "observation_count",
        "rate_per_100",
        "species_breakdown",
    ):
        assert key in feature
