"""Tests for the /api/sites endpoints."""

from tests.conftest import make_project, make_site


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
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Site A"
    assert data["project_id"] == p.id


def test_create_site_invalid_project(client):
    resp = client.post("/api/sites", json={
        "name": "Site A",
        "project_id": "nonexistent",
    })
    assert resp.status_code == 400


def test_create_site_duplicate_name(client, db):
    p = make_project(db)
    make_site(db, project_id=p.id, name="dup")
    resp = client.post("/api/sites", json={
        "name": "dup",
        "project_id": p.id,
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
