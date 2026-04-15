"""Tests for the /api/events endpoints."""

from datetime import datetime
from unittest.mock import patch

from tests.conftest import (
    make_deployment,
    make_event_with_files,
    make_project,
    make_site,
)


def test_generate_events(client, db):
    p = make_project(db)
    with patch(
        "app.api.routers.events.event_crud.generate_events_for_project",
        return_value=5,
    ):
        resp = client.post("/api/events/generate", json={"project_id": p.id})
    assert resp.status_code == 200
    assert resp.json()["event_count"] == 5


def test_generate_events_project_not_found(client):
    with patch(
        "app.api.routers.events.event_crud.generate_events_for_project",
        side_effect=ValueError("Project not found"),
    ):
        resp = client.post("/api/events/generate", json={"project_id": "nonexistent"})
    assert resp.status_code == 404


def test_list_events_empty(client, db):
    p = make_project(db)
    resp = client.get(f"/api/events?project_id={p.id}")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_event_count(client, db):
    p = make_project(db)
    resp = client.get(f"/api/events/count?project_id={p.id}")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_get_verification_stats(client, db):
    p = make_project(db)
    resp = client.get(f"/api/events/verification-stats?project_id={p.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_files" in data
    assert "verified_files" in data


def test_get_event_not_found(client):
    resp = client.get("/api/events/nonexistent")
    assert resp.status_code == 404


def test_get_event_with_files(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = make_event_with_files(
        db,
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    resp = client.get(f"/api/events/{ev.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == ev.id
    assert len(data["files"]) == 1


def test_get_adjacent_events(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = make_event_with_files(
        db,
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    resp = client.get(
        f"/api/events/{ev.id}/adjacent?project_id={p.id}"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "previous_id" in data
    assert "next_id" in data
