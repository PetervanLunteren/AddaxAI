"""Tests for the /api/jobs endpoints."""

from unittest.mock import patch

from tests.conftest import make_job, make_project


def test_list_jobs_empty(client):
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_job(client):
    resp = client.post("/api/jobs", json={
        "type": "deployment_analysis",
        "payload": {"project_id": "test"},
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "deployment_analysis"
    assert data["status"] == "pending"


def test_get_job(client, db):
    j = make_job(db)
    resp = client.get(f"/api/jobs/{j.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == j.id


def test_get_job_not_found(client):
    resp = client.get("/api/jobs/nonexistent")
    assert resp.status_code == 404


def test_update_job(client, db):
    j = make_job(db)
    resp = client.patch(f"/api/jobs/{j.id}", json={"status": "running"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_update_job_not_found(client):
    resp = client.patch("/api/jobs/nonexistent", json={"status": "running"})
    assert resp.status_code == 404


def test_delete_job(client, db):
    j = make_job(db)
    resp = client.delete(f"/api/jobs/{j.id}")
    assert resp.status_code == 204


def test_delete_job_not_found(client):
    resp = client.delete("/api/jobs/nonexistent")
    assert resp.status_code == 404


def test_run_queue_no_pending(client):
    resp = client.post("/api/jobs/run-queue")
    assert resp.status_code == 200
    assert resp.json()["jobs_started"] == 0


def test_run_queue_with_pending(client, db):
    make_job(db, payload={"project_id": "p1"})
    with patch("app.api.routers.jobs.ws_manager"):
        resp = client.post("/api/jobs/run-queue?project_id=p1")
    assert resp.status_code == 200
    assert resp.json()["jobs_started"] >= 1


def test_reconcile_interrupted_jobs(db):
    """
    Startup reconciliation fails jobs left `running` by a previous process
    and resets their stuck `processing` queue entries, leaving pending and
    already-terminal rows untouched.
    """
    from app.api.crud import deployment_queue as crud_queue
    from app.api.crud.job import reconcile_interrupted_jobs
    from app.api.schemas.deployment_queue import DeploymentQueueCreate

    running = make_job(db, status="running")
    pending = make_job(db, status="pending")
    completed = make_job(db, status="completed")

    project = make_project(db)
    processing = crud_queue.create_queue_entry(
        db, DeploymentQueueCreate(project_id=project.id, folder_path="/x")
    )
    crud_queue.update_queue_status(db, processing.id, status="processing")
    pending_entry = crud_queue.create_queue_entry(
        db, DeploymentQueueCreate(project_id=project.id, folder_path="/y")
    )

    assert reconcile_interrupted_jobs(db) == 1

    for job in (running, pending, completed):
        db.refresh(job)
    assert running.status == "failed"
    assert running.error
    assert running.completed_at_utc is not None
    assert pending.status == "pending"
    assert completed.status == "completed"

    db.refresh(processing)
    db.refresh(pending_entry)
    assert processing.status == "failed"
    assert processing.error
    assert pending_entry.status == "pending"
