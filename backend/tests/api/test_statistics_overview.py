"""Tests for the dashboard overview date filtering and date validation.

Pins the two boundary rules every dashboard stat now shares:
- the event and observation summary cards respect the date window
- date_to is inclusive of the whole end day (files captured late on
  the end date still count)
and the router-level 422 on malformed date params.
"""

from datetime import datetime

from app.api.crud import statistics as stats_crud
from app.models.event_observation import EventObservation
from tests.conftest import (
    make_deployment,
    make_event_with_files,
    make_project,
    make_site,
)


def _add_observation(db, *, event_id: str, label: str, max_n: int = 1):
    obs = EventObservation(
        event_id=event_id,
        label=label,
        category="animal",
        max_n=max_n,
    )
    db.add(obs)
    db.flush()
    return obs


def _build_fixture(db):
    """Two events in January, one in February, each with one file and
    one observation. The February observation has max_n=5 so the
    observation totals differ clearly per window."""
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)

    ev_jan1 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 2, 8, 0)
    )
    # Late on Jan 31: only counted if date_to=2024-01-31 includes the
    # whole end day.
    ev_jan2 = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 31, 22, 30)
    )
    ev_feb = make_event_with_files(
        db, deployment_id=dep.id, event_start_local=datetime(2024, 2, 10, 9, 0)
    )

    _add_observation(db, event_id=ev_jan1.id, label="leopard")
    _add_observation(db, event_id=ev_jan2.id, label="lion")
    _add_observation(db, event_id=ev_feb.id, label="leopard", max_n=5)

    db.flush()
    return project


def test_overview_counts_respect_date_range(db):
    project = _build_fixture(db)

    january = stats_crud.get_dashboard_overview(
        db, project.id, date_from="2024-01-01", date_to="2024-01-31"
    )
    assert january.total_events == 2
    assert january.total_observations == 2
    assert january.total_files == 2

    february = stats_crud.get_dashboard_overview(
        db, project.id, date_from="2024-02-01", date_to="2024-02-29"
    )
    assert february.total_events == 1
    assert february.total_observations == 5
    assert february.total_files == 1


def test_overview_unfiltered_counts_everything(db):
    project = _build_fixture(db)

    overview = stats_crud.get_dashboard_overview(db, project.id)
    assert overview.total_events == 3
    assert overview.total_observations == 7
    assert overview.total_files == 3


def test_date_to_includes_whole_end_day(db):
    """A file captured at 22:30 on the end date counts. The old string
    comparison ('2024-01-31 22:30:00' <= '2024-01-31') dropped it."""
    project = _build_fixture(db)

    single_day = stats_crud.get_dashboard_overview(
        db, project.id, date_from="2024-01-31", date_to="2024-01-31"
    )
    assert single_day.total_files == 1
    assert single_day.total_events == 1
    assert single_day.total_observations == 1


def test_malformed_date_params_return_422(client, db):
    project = make_project(db)

    for endpoint in ("overview", "species", "detection-trend"):
        resp = client.get(
            f"/api/statistics/{endpoint}",
            params={"project_id": project.id, "date_from": "banana"},
        )
        assert resp.status_code == 422, endpoint
        assert "date_from" in resp.json()["detail"]

    resp = client.get(
        "/api/statistics/overview",
        params={"project_id": project.id, "date_to": "2024-13-99"},
    )
    assert resp.status_code == 422
    assert "date_to" in resp.json()["detail"]


def test_valid_date_params_still_accepted(client, db):
    project = _build_fixture(db)

    resp = client.get(
        "/api/statistics/overview",
        params={
            "project_id": project.id,
            "date_from": "2024-01-01",
            "date_to": "2024-01-31",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["total_events"] == 2
