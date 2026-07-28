"""
End-to-end wire-format tests for observational datetime serialization.

Guards against the subtle bug where a ContextVar set inside a sync
FastAPI endpoint (running in a threadpool) isn't visible to FastAPI's
response serialization stage (running in the event loop task). These
tests hit the real endpoint via TestClient and assert that the JSON
wire format carries the project's tz offset.
"""

from datetime import datetime

from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def test_get_event_serializes_captured_at_local_with_amsterdam_winter_offset(client, db):
    """
    Camera wall-clock 08:25 on a January date in an Amsterdam-tz project
    must come out as `...T08:25:00+01:00`, not `...T08:25:00+00:00` or
    `...T08:25:00`. The offset is the whole point: without it, the
    browser would interpret the timestamp in the viewer's local tz and
    display a wrong hour (the bug this test pins down).
    """
    project = make_project(db, timezone="Europe/Amsterdam")
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev = make_event_with_files(
        db,
        deployment_id=dep.id,
        event_start_local=datetime(2013, 1, 26, 8, 25, 0),
    )
    db.flush()

    resp = client.get(f"/api/events/{ev.id}")
    assert resp.status_code == 200
    data = resp.json()

    # Outer event fields use the project tz on the event's local date.
    assert data["event_start_local"] == "2013-01-26T08:25:00+01:00"
    assert data["event_end_local"] == "2013-01-26T08:25:00+01:00"

    # Nested files also carry the offset. Without the async fix, these
    # serialize with "+00:00" because the sync endpoint's ContextVar
    # change is lost before FastAPI runs jsonable_encoder.
    assert len(data["files"]) == 1
    assert data["files"][0]["captured_at_local"] == "2013-01-26T08:25:00+01:00"


def test_get_event_serializes_amsterdam_summer_offset(client, db):
    """Same zone, summer date → CEST, +02:00."""
    project = make_project(db, timezone="Europe/Amsterdam")
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev = make_event_with_files(
        db,
        deployment_id=dep.id,
        event_start_local=datetime(2024, 6, 15, 7, 30, 0),
    )
    db.flush()

    resp = client.get(f"/api/events/{ev.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["event_start_local"] == "2024-06-15T07:30:00+02:00"
    assert data["files"][0]["captured_at_local"] == "2024-06-15T07:30:00+02:00"


def test_list_events_serializes_with_project_offset(client, db):
    """The /api/events list endpoint also needs the async fix.

    Note: we attach a high-confidence detection because list_events
    applies the project's detection threshold via an EXISTS subquery
    over the event's files. Without a passing detection the event is
    filtered out and we'd be testing an empty list.
    """
    project = make_project(db, timezone="Europe/Amsterdam")
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev = make_event_with_files(
        db,
        deployment_id=dep.id,
        event_start_local=datetime(2013, 1, 26, 8, 25, 0),
    )
    file_id = ev.files[0].id
    make_detection(db, file_id=file_id, confidence=0.95)
    db.flush()

    resp = client.get(f"/api/events?project_id={project.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["event_start_local"] == "2013-01-26T08:25:00+01:00"


def test_get_file_serializes_with_project_offset(client, db):
    """The /api/files/{id} endpoint also needs the async fix."""
    project = make_project(db, timezone="Europe/Amsterdam")
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    f = make_file(
        db,
        deployment_id=dep.id,
        captured_at_local=datetime(2013, 1, 26, 8, 25, 0),
    )
    make_detection(db, file_id=f.id)
    db.flush()

    resp = client.get(f"/api/files/{f.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["captured_at_local"] == "2013-01-26T08:25:00+01:00"


def test_fixed_offset_zone_has_no_dst_jump(client, db):
    """
    Cameras configured to a fixed UTC offset (no DST) must serialize
    the same offset in both summer and winter. Uses Etc/GMT-3 (UTC+3;
    IANA inverts the sign). Covers the "local winter time" use case
    where users pick a fixed offset so the clock never jumps.
    """
    project = make_project(db, timezone="Etc/GMT-3")
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev_winter = make_event_with_files(
        db,
        deployment_id=dep.id,
        event_start_local=datetime(2024, 1, 15, 10, 0, 0),
    )
    ev_summer = make_event_with_files(
        db,
        deployment_id=dep.id,
        event_start_local=datetime(2024, 7, 15, 10, 0, 0),
    )
    db.flush()

    resp_w = client.get(f"/api/events/{ev_winter.id}")
    resp_s = client.get(f"/api/events/{ev_summer.id}")
    assert resp_w.status_code == 200
    assert resp_s.status_code == 200
    assert resp_w.json()["event_start_local"] == "2024-01-15T10:00:00+03:00"
    assert resp_s.json()["event_start_local"] == "2024-07-15T10:00:00+03:00"
