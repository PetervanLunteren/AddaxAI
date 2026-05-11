"""Tests for the observation-rate-map endpoint and its CRUD function.

Each map feature represents one camera site, aggregating across all of
the site's deployments. The fixture sets up two sites with one
deployment each; a separate test covers the multi-deployment-per-site
aggregation path.
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
    """One project, two sites, one deployment each, a few events with MaxN."""
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
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 1, 11),  # 10 nights
    )
    dep_b = make_deployment(
        db,
        site_id=site_b.id,
        start_date_local=date(2024, 2, 1),
        end_date_local=date(2024, 2, 6),  # 5 nights
    )

    # Two events at deployment A
    ev_a1 = make_event_with_files(
        db, deployment_id=dep_a.id, event_start_local=datetime(2024, 1, 2, 8, 0)
    )
    ev_a2 = make_event_with_files(
        db, deployment_id=dep_a.id, event_start_local=datetime(2024, 1, 5, 14, 0)
    )

    # One event at deployment B
    ev_b1 = make_event_with_files(
        db, deployment_id=dep_b.id, event_start_local=datetime(2024, 2, 3, 9, 0)
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
    """Trap nights is folder-aware and driven by actual file captures.
    dep_a has files at Jan 2 and Jan 5 (same folder) -> 4 inclusive days.
    dep_b has a single file at Feb 3 -> 1 day. Total 5."""
    project, _, _, dep_a, dep_b = _build_fixture(db)

    per_dep = stats_crud.get_per_deployment_trap_nights(db, project.id)
    assert per_dep[dep_a.id] == 4
    assert per_dep[dep_b.id] == 1

    total = stats_crud.get_trap_nights(db, project.id)
    assert total == 5


# ---------------------------------------------------------------------------
# get_observation_rate_map
# ---------------------------------------------------------------------------


def test_observation_rate_map_returns_one_feature_per_site(db):
    project, site_a, site_b, _, _ = _build_fixture(db)

    response = stats_crud.get_observation_rate_map(db, project.id)
    assert len(response.features) == 2

    by_id = {f.site_id: f for f in response.features}
    feature_a = by_id[site_a.id]
    assert feature_a.site_name == "Alpha"
    assert feature_a.latitude == 10.0
    assert feature_a.longitude == 20.0
    assert feature_a.deployment_count == 1
    # Folder-aware trap nights: Jan 2 and Jan 5 in the same folder
    # -> 4 inclusive days. Rate = 5 obs / 4 nights * 100.
    assert feature_a.trap_nights == 4
    assert feature_a.observation_count == 5  # 2 + 3
    assert feature_a.rate_per_100 == 125.0
    assert feature_a.earliest_start_local == date(2024, 1, 1)
    assert feature_a.latest_end_local == date(2024, 1, 11)

    feature_b = by_id[site_b.id]
    assert feature_b.site_name == "Beta"
    assert feature_b.deployment_count == 1
    # Single file at Feb 3 -> 1 trap night. Rate = 1 / 1 * 100.
    assert feature_b.trap_nights == 1
    assert feature_b.observation_count == 1
    assert feature_b.rate_per_100 == 100.0


def test_observation_rate_map_filters_by_site(db):
    project, site_a, _, _, _ = _build_fixture(db)

    response = stats_crud.get_observation_rate_map(
        db, project.id, site_ids=[site_a.id]
    )
    assert len(response.features) == 1
    assert response.features[0].site_id == site_a.id


def test_observation_rate_map_filters_by_date(db):
    project, site_a, site_b, _, _ = _build_fixture(db)

    # Date range that only catches site A's events.
    response = stats_crud.get_observation_rate_map(
        db, project.id, date_from="2024-01-01", date_to="2024-01-31"
    )
    by_id = {f.site_id: f for f in response.features}

    # Site A: events in range, rate computed on clipped trap nights.
    assert site_a.id in by_id
    assert by_id[site_a.id].observation_count == 5

    # Site B: its deployment falls after the filter window, trap nights
    # clip to 0, no observations -> feature dropped.
    assert site_b.id not in by_id


def test_observation_rate_map_includes_species_breakdown(db):
    project, site_a, site_b, _, _ = _build_fixture(db)

    response = stats_crud.get_observation_rate_map(db, project.id)
    by_id = {f.site_id: f for f in response.features}

    leopard = by_id[site_a.id].species_breakdown
    assert len(leopard) == 1
    assert leopard[0].label == "leopard"
    assert leopard[0].count == 5

    lion = by_id[site_b.id].species_breakdown
    assert len(lion) == 1
    assert lion[0].label == "lion"
    assert lion[0].count == 1


def test_observation_rate_map_aggregates_multiple_deployments_per_site(db):
    """One site with two deployments at disjoint windows produces a
    single feature with summed nights, summed observations, merged
    species breakdown, and a date range spanning both."""
    project = make_project(db)
    site = make_site(
        db, project_id=project.id, name="Camp 1", latitude=5.0, longitude=15.0
    )

    dep_1 = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 3, 1),
        end_date_local=date(2024, 3, 10),
    )
    dep_2 = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 6, 1),
        end_date_local=date(2024, 6, 7),
    )

    ev_1 = make_event_with_files(
        db, deployment_id=dep_1.id, event_start_local=datetime(2024, 3, 2, 10, 0)
    )
    ev_2 = make_event_with_files(
        db, deployment_id=dep_1.id, event_start_local=datetime(2024, 3, 8, 11, 0)
    )
    ev_3 = make_event_with_files(
        db, deployment_id=dep_2.id, event_start_local=datetime(2024, 6, 4, 12, 0)
    )

    _add_observation(db, event_id=ev_1.id, label="leopard", max_n=1)
    _add_observation(db, event_id=ev_2.id, label="hyena", max_n=2)
    _add_observation(db, event_id=ev_3.id, label="leopard", max_n=4)
    db.flush()

    response = stats_crud.get_observation_rate_map(db, project.id)
    assert len(response.features) == 1

    feature = response.features[0]
    assert feature.site_id == site.id
    assert feature.deployment_count == 2

    # dep_1 has files at Mar 2 and Mar 8 (same folder) -> 7 inclusive days.
    # dep_2 has a single file at Jun 4 -> 1 day. Total 8.
    assert feature.trap_nights == 8
    # 1 + 2 + 4 = 7 across both deployments.
    assert feature.observation_count == 7
    assert feature.rate_per_100 == 7 / 8 * 100

    # Range spans both deployments, not clipped to a filter window.
    assert feature.earliest_start_local == date(2024, 3, 1)
    assert feature.latest_end_local == date(2024, 6, 7)

    # Species breakdown merges labels across both deployments. Leopard
    # appears in both (1 + 4 = 5); hyena only in dep_1 (2).
    by_label = {row.label: row.count for row in feature.species_breakdown}
    assert by_label["leopard"] == 5
    assert by_label["hyena"] == 2


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
        "site_id",
        "site_name",
        "latitude",
        "longitude",
        "deployment_count",
        "earliest_start_local",
        "latest_end_local",
        "trap_nights",
        "observation_count",
        "rate_per_100",
        "species_breakdown",
    ):
        assert key in feature
    assert "deployment_id" not in feature
    assert "start_date_local" not in feature
    assert "end_date_local" not in feature
