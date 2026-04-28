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


def _setup_three_events(db):
    """Three events with distinct start times for sort tests."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    older = _event_with_detection(db, d.id, datetime(2024, 1, 1, 12, 0))
    middle = _event_with_detection(db, d.id, datetime(2024, 2, 1, 12, 0))
    newer = _event_with_detection(db, d.id, datetime(2024, 3, 1, 12, 0))
    return p, older, middle, newer


def test_events_sort_oldest(client, db):
    p, older, middle, newer = _setup_three_events(db)

    resp = client.get(f"/api/events?project_id={p.id}&sort=oldest")
    ids = [row["id"] for row in resp.json()]
    assert ids == [older.id, middle.id, newer.id]


def test_events_sort_random_stable_with_seed(client, db):
    p, *_ = _setup_three_events(db)

    a = client.get(f"/api/events?project_id={p.id}&sort=random&seed=42").json()
    b = client.get(f"/api/events?project_id={p.id}&sort=random&seed=42").json()
    c = client.get(f"/api/events?project_id={p.id}&sort=random&seed=99").json()
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["id"] for r in a] != [r["id"] for r in c]


def test_events_sort_cls_low_pushes_nulls_last(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    low = make_event_with_files(
        db, deployment_id=d.id, event_start_local=datetime(2024, 1, 1, 12, 0),
    )
    high = make_event_with_files(
        db, deployment_id=d.id, event_start_local=datetime(2024, 2, 1, 12, 0),
    )
    null = make_event_with_files(
        db, deployment_id=d.id, event_start_local=datetime(2024, 3, 1, 12, 0),
    )
    make_detection(
        db, file_id=low.files[0].id, confidence=0.9, label_confidence=0.2,
    )
    make_detection(
        db, file_id=high.files[0].id, confidence=0.9, label_confidence=0.7,
    )
    make_detection(
        db, file_id=null.files[0].id, confidence=0.9, label_confidence=None,
    )
    db.commit()

    resp = client.get(f"/api/events?project_id={p.id}&sort=cls_low")
    ids = [row["id"] for row in resp.json()]
    assert ids == [low.id, high.id, null.id]


def test_events_adjacent_respects_sort(client, db):
    p, older, middle, newer = _setup_three_events(db)

    # Oldest-first: opening "older" → next is "middle".
    resp = client.get(
        f"/api/events/{older.id}/adjacent?project_id={p.id}&sort=oldest"
    ).json()
    assert resp["next_id"] == middle.id
    assert resp["previous_id"] is None

    # Newest-first (default): opening "older" → next is None.
    resp = client.get(
        f"/api/events/{older.id}/adjacent?project_id={p.id}"
    ).json()
    assert resp["next_id"] is None
    assert resp["previous_id"] == middle.id


def test_events_sort_invalid_value_returns_400(client, db):
    p = make_project(db)
    db.commit()

    resp = client.get(f"/api/events?project_id={p.id}&sort=bogus")
    assert resp.status_code == 400
