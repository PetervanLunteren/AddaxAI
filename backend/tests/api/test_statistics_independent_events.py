"""Every statistic counts independent events, not stored sequences.

Through the API, so the routers, the cache and the queries are all on the
path. The rule itself is pinned in test_independent_events.py; these check
that each statistic reads it.
"""

from datetime import datetime, timedelta

import pytest

from app.models.sequence import Sequence
from app.models.sequence_observation import SequenceObservation
from tests.conftest import make_deployment, make_project, make_site

T0 = datetime(2024, 6, 30, 23, 40, 0)


def _sequence(db, dep, minutes, deer, *, camera=""):
    start = T0 + timedelta(minutes=minutes)
    s = Sequence(
        deployment_id=dep.id, sequence_start_local=start, sequence_end_local=start,
        file_count=1, camera_key=camera,
    )
    db.add(s)
    db.flush()
    db.add(SequenceObservation(sequence_id=s.id, label="deer", category="animal", max_n=deer))
    db.flush()
    return s


@pytest.fixture()
def lingering_deer(db):
    """Ten sequences of deer three minutes apart, from 23:40 on 30 June into
    1 July: nine of 5 deer and one of 10. One independent event of 10 at the
    default 30-minute interval; ten stored sequences summing 55."""
    project = make_project(db, timezone="UTC")
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id, project_id=project.id)
    for i in range(10):
        _sequence(db, dep, i * 3, 10 if i == 4 else 5)
    db.commit()
    return project, site, dep


def _summary(client, project_id, **params):
    r = client.get("/api/statistics/summary", params={"project_id": project_id, **params})
    assert r.status_code == 200, r.text
    return r.json()


def test_the_dashboard_counts_one_event_with_its_highest_count(client, lingering_deer):
    project, *_ = lingering_deer
    summary = _summary(client, project.id)
    assert (summary["events"], summary["observations"]) == (1, 10)


def test_the_species_bars_count_events_and_their_counts(client, lingering_deer):
    project, *_ = lingering_deer
    for mode, expected in (("events", 1), ("max_n", 10)):
        r = client.get("/api/statistics/species", params={
            "project_id": project.id, "count_mode": mode,
        })
        assert [(row["species"], row["count"]) for row in r.json()] == [("deer", expected)]


def test_a_date_window_keeps_an_event_by_its_start(client, lingering_deer):
    """The event starts on 30 June and runs past midnight: it belongs to June
    only, so it never counts in two windows."""
    project, *_ = lingering_deer
    june = _summary(client, project.id, date_from="2024-06-01", date_to="2024-06-30")
    july = _summary(client, project.id, date_from="2024-07-01", date_to="2024-07-31")
    assert (june["events"], june["observations"]) == (1, 10)
    assert (july["events"], july["observations"]) == (0, 0)


def test_the_activity_clock_puts_the_event_at_its_start(client, lingering_deer):
    project, *_ = lingering_deer
    r = client.get("/api/statistics/activity-pattern", params={"project_id": project.id})
    counts = {h["hour"]: h["count"] for h in r.json()["hours"] if h["count"]}
    assert counts == {23: 10}


def test_the_sheets_count_events_and_keep_the_stored_sequences(client, lingering_deer):
    """Observations follow the events; the number of stored sequences is a
    count of cards, so it stays what is stored."""
    project, site, dep = lingering_deer
    for url in (f"/api/deployments/{dep.id}/info", f"/api/sites/{site.id}/info"):
        info = client.get(url).json()
        assert info["observation_count"] == 10, url
        assert info["sequence_count"] == 10, url
    projects = client.get("/api/projects", params={"mode": "research"}).json()
    assert [p["observation_count"] for p in projects if p["id"] == project.id] == [10]


def test_a_new_interval_shows_at_once_without_a_reprocess(client, db, lingering_deer):
    """Three minutes apart: at a 1-minute interval every sequence is an
    event of its own. Nothing is regrouped and no reprocess is needed."""
    project, *_ = lingering_deer
    r = client.patch(f"/api/projects/{project.id}", json={"independence_interval": 60})
    assert r.status_code == 200, r.text
    db.commit()
    summary = _summary(client, project.id)
    assert (summary["events"], summary["observations"]) == (10, 55)
    status = client.get(f"/api/projects/{project.id}/postprocessing-status").json()
    assert status["needs_reprocessing"] is False


def test_cameras_of_one_deployment_never_join(client, db):
    project = make_project(db, timezone="UTC")
    dep = make_deployment(db, project_id=project.id)
    _sequence(db, dep, 0, 2, camera="/d/a/")
    _sequence(db, dep, 1, 3, camera="/d/b/")
    db.commit()
    summary = _summary(client, project.id)
    assert (summary["events"], summary["observations"]) == (2, 5)
