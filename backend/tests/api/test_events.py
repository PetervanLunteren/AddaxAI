"""Tests for the /api/events endpoints."""

from datetime import datetime
from unittest.mock import patch

from tests.conftest import (
    make_deployment,
    make_detection,
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


def _event_with_detection(db, deployment_id, start):
    """Helper: event + one file + one visible detection so the project
    threshold filter doesn't drop the event."""
    ev = make_event_with_files(
        db,
        deployment_id=deployment_id,
        event_start_local=start,
    )
    make_detection(db, file_id=ev.files[0].id, confidence=0.9)
    db.commit()
    return ev


def test_events_filter_flagged_exists_on_any_file(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev_with_flag = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    ev_without = _event_with_detection(db, d.id, datetime(2024, 1, 2, 12, 0))
    flagged_file = ev_with_flag.files[0]
    client.patch(f"/api/files/{flagged_file.id}", json={"flagged": True})

    resp = client.get(f"/api/events?project_id={p.id}&flagged=flagged")
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()]
    assert ev_with_flag.id in ids
    assert ev_without.id not in ids


def test_events_filter_favorited_exists_on_any_file(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev_fav = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    ev_none = _event_with_detection(db, d.id, datetime(2024, 1, 2, 12, 0))
    fav_file = ev_fav.files[0]
    client.patch(f"/api/files/{fav_file.id}", json={"favorited": True})

    resp = client.get(f"/api/events?project_id={p.id}&favorited=favorited")
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()]
    assert ev_fav.id in ids
    assert ev_none.id not in ids


def test_events_filter_label_confidence_range(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev_high = make_event_with_files(
        db, deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    ev_low = make_event_with_files(
        db, deployment_id=d.id,
        event_start_local=datetime(2024, 1, 2, 12, 0),
    )
    make_detection(
        db, file_id=ev_high.files[0].id, confidence=0.9, label_confidence=0.9,
    )
    make_detection(
        db, file_id=ev_low.files[0].id, confidence=0.9, label_confidence=0.3,
    )
    db.commit()

    resp = client.get(
        f"/api/events?project_id={p.id}&min_label_confidence=0.5"
    )
    ids = [row["id"] for row in resp.json()]
    assert ev_high.id in ids
    assert ev_low.id not in ids


def test_events_filter_label_confidence_excludes_null(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    classified = make_event_with_files(
        db, deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    unclassified = make_event_with_files(
        db, deployment_id=d.id,
        event_start_local=datetime(2024, 1, 2, 12, 0),
    )
    make_detection(
        db, file_id=classified.files[0].id, confidence=0.9, label_confidence=0.3,
    )
    make_detection(
        db, file_id=unclassified.files[0].id, confidence=0.9, label_confidence=None,
    )
    db.commit()

    resp = client.get(
        f"/api/events?project_id={p.id}&min_label_confidence=0.0"
    )
    ids = [row["id"] for row in resp.json()]
    assert classified.id in ids
    assert unclassified.id not in ids


def test_events_filter_empty_show_only_and_hide(client, db):
    """Empty event = every file in it has observation_type='blank'."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev_animal = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    ev_blank = _event_with_detection(db, d.id, datetime(2024, 1, 2, 12, 0))
    # Mark all files in ev_blank as observation_type='blank'.
    for f in ev_blank.files:
        f.observation_type = "blank"
    db.commit()

    resp = client.get(f"/api/events?project_id={p.id}&empty=show_only")
    ids = [row["id"] for row in resp.json()]
    assert ev_blank.id in ids
    assert ev_animal.id not in ids

    resp = client.get(f"/api/events?project_id={p.id}&empty=hide")
    ids = [row["id"] for row in resp.json()]
    assert ev_animal.id in ids
    assert ev_blank.id not in ids


def test_event_summary_aggregates_any_file_flagged_and_favorited(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    f = ev.files[0]
    client.patch(f"/api/files/{f.id}", json={"flagged": True, "favorited": True})

    resp = client.get(f"/api/events?project_id={p.id}")
    assert resp.status_code == 200
    summary = next(row for row in resp.json() if row["id"] == ev.id)
    assert summary["any_file_flagged"] is True
    assert summary["any_file_favorited"] is True
