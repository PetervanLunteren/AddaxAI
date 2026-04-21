"""Tests for the /api/sites endpoints."""

from datetime import date, datetime

import pytest

from app.models.event_observation import EventObservation
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def test_list_sites_empty(client):
    resp = client.get("/api/sites")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_sites_filtered_by_project(client, db):
    p1 = make_project(db)
    p2 = make_project(db)
    make_site(db, project_id=p1.id)
    make_site(db, project_id=p2.id)
    resp = client.get(f"/api/sites?project_id={p1.id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_create_site(client, db):
    p = make_project(db)
    resp = client.post("/api/sites", json={
        "name": "Site A",
        "project_id": p.id,
        "latitude": 52.0,
        "longitude": 5.0,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Site A"
    assert data["project_id"] == p.id


def test_create_site_invalid_project(client):
    resp = client.post("/api/sites", json={
        "name": "Site A",
        "project_id": "nonexistent",
        "latitude": 52.0,
        "longitude": 5.0,
    })
    assert resp.status_code == 400


def test_create_site_missing_gps(client, db):
    """GPS coordinates are now required."""
    p = make_project(db)
    resp = client.post("/api/sites", json={
        "name": "Site A",
        "project_id": p.id,
    })
    assert resp.status_code == 422


def test_create_site_duplicate_name(client, db):
    p = make_project(db)
    make_site(db, project_id=p.id, name="dup")
    resp = client.post("/api/sites", json={
        "name": "dup",
        "project_id": p.id,
        "latitude": 52.0,
        "longitude": 5.0,
    })
    assert resp.status_code == 409


def test_get_site(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    resp = client.get(f"/api/sites/{s.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == s.id


def test_get_site_not_found(client):
    resp = client.get("/api/sites/nonexistent")
    assert resp.status_code == 404


def test_update_site(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    resp = client.patch(f"/api/sites/{s.id}", json={"name": "Updated"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated"


def test_update_site_not_found(client):
    resp = client.patch("/api/sites/nonexistent", json={"name": "x"})
    assert resp.status_code == 404


def test_delete_site(client, db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    resp = client.delete(f"/api/sites/{s.id}")
    assert resp.status_code == 204


def test_delete_site_not_found(client):
    resp = client.delete("/api/sites/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /info endpoint
# ---------------------------------------------------------------------------


def _build_site_info_fixture(db):
    """Site with 2 deployments, mixed file types, a classified
    detection, a verified below-threshold detection, and one event +
    observation. Used by the happy-path test."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(
        db,
        project_id=project.id,
        name="Nordmarka",
        latitude=59.91,
        longitude=10.75,
        elevation_m=200.0,
        habitat_type="forest",
    )
    # Two deployments with end dates so trap_nights aggregates cleanly.
    dep1 = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 6, 1),
        end_date_local=date(2024, 6, 10),  # 10 nights
    )
    dep2 = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 7, 1),
        end_date_local=date(2024, 7, 5),  # 5 nights
    )
    # Files: 3 images + 1 video on dep1, 1 image on dep2 = 4 images + 1 video.
    for i in range(3):
        make_file(
            db,
            deployment_id=dep1.id,
            file_type="image",
            file_format="jpg",
            captured_at_local=datetime(2024, 6, 5, 8, i),
        )
    make_file(
        db,
        deployment_id=dep1.id,
        file_type="video",
        file_format="mp4",
        captured_at_local=datetime(2024, 6, 5, 9, 0),
    )
    file_dep2 = make_file(
        db,
        deployment_id=dep2.id,
        file_type="image",
        file_format="jpg",
        captured_at_local=datetime(2024, 7, 3, 14, 0),
    )
    # One classified detection (above threshold) + one verified
    # below-threshold detection on dep2.
    make_detection(
        db,
        file_id=file_dep2.id,
        confidence=0.9,
        label="lion",
        label_confidence=0.8,
    )
    make_detection(
        db,
        file_id=file_dep2.id,
        confidence=0.2,
        verified=True,
    )
    # One event with one observation (MaxN=2) on dep1.
    event = make_event_with_files(
        db,
        deployment_id=dep1.id,
        event_start_local=datetime(2024, 6, 5, 8, 0),
        files_verified=[],  # suppress auto-created file
    )
    db.add(
        EventObservation(
            event_id=event.id,
            label="lion",
            label_taxonomy_id=None,
            category="animal",
            max_n=2,
        )
    )
    db.flush()
    return site


def test_site_info_happy_path(client, db):
    site = _build_site_info_fixture(db)
    resp = client.get(f"/api/sites/{site.id}/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_id"] == site.id
    assert data["name"] == "Nordmarka"
    assert data["latitude"] == pytest.approx(59.91)
    assert data["longitude"] == pytest.approx(10.75)
    assert data["elevation_m"] == pytest.approx(200.0)
    assert data["habitat_type"] == "forest"
    assert data["deployment_count"] == 2
    assert data["files"] == {"total": 5, "images": 4, "videos": 1}
    assert data["event_count"] == 1
    assert data["observation_count"] == 2
    assert data["detection_categories"]["animal"] == 2
    assert data["detection_categories"]["person"] == 0
    assert data["top_species"] == [
        {"label": "lion", "display_name": None, "count": 2}
    ]
    # Trap nights is folder-aware (sum per deployment of per-folder
    # (max - min + 1) over actual captures). dep1's 4 files all fall on
    # June 5 → 1 night. dep2's single file is July 3 → 1 night. Total 2.
    # The manually-set Deployment.start_date_local / end_date_local are
    # ignored now (they're display-only).
    assert data["trap_nights"] == 2
    # Rate: 2 obs / 2 nights * 100 = 100.
    assert data["observation_rate_per_100_trap_nights"] == pytest.approx(100.0)
    assert data["verification"] == {"verified": 0, "total": 5}


def test_site_info_not_found(client):
    resp = client.get("/api/sites/nonexistent/info")
    assert resp.status_code == 404


def test_site_info_empty_site(client, db):
    project = make_project(db, detection_threshold=0.5)
    site = make_site(
        db, project_id=project.id, name="Empty", latitude=0.0, longitude=0.0
    )
    resp = client.get(f"/api/sites/{site.id}/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deployment_count"] == 0
    assert data["files"] == {"total": 0, "images": 0, "videos": 0}
    assert data["event_count"] == 0
    assert data["observation_count"] == 0
    assert data["trap_nights"] is None
    assert data["observation_rate_per_100_trap_nights"] is None
    assert data["top_species"] == []
    assert data["first_captured_at_local"] is None
    assert data["last_captured_at_local"] is None


def test_site_info_trap_nights_from_file_spans(client, db):
    """Trap nights is folder-aware: each deployment contributes the sum
    of per-folder (max - min + 1) day spans over its actual files. A
    deployment with no files contributes 0. The manually-set
    start_date_local / end_date_local on the Deployment row are no
    longer consulted for this calculation."""
    project = make_project(db, detection_threshold=0.5)
    site = make_site(
        db, project_id=project.id, name="Mixed", latitude=0.0, longitude=0.0
    )
    dep_with_files = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 6, 1),
        end_date_local=date(2024, 6, 10),  # display-only
    )
    dep_empty = make_deployment(
        db,
        site_id=site.id,
        start_date_local=date(2024, 7, 1),
        end_date_local=None,
    )
    # 3 files spanning June 5 .. June 7 = 3 inclusive days.
    for day in (5, 6, 7):
        make_file(
            db,
            deployment_id=dep_with_files.id,
            file_type="image",
            file_format="jpg",
            captured_at_local=datetime(2024, 6, day, 12, 0),
        )
    # dep_empty has no files; contributes 0.
    _ = dep_empty

    resp = client.get(f"/api/sites/{site.id}/info")
    data = resp.json()
    assert data["deployment_count"] == 2
    assert data["trap_nights"] == 3
