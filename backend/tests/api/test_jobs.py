"""Tests for the /api/jobs endpoints."""

from unittest.mock import patch

from sqlalchemy.orm import sessionmaker

from app.models import Job
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
    Startup reconciliation fails jobs left `running` OR `pending` by a
    previous process and resets their stuck `processing` queue entries,
    leaving already-terminal rows untouched.

    `pending` counts as orphaned because a job only starts when the
    frontend sends "ready" and `ws_manager` holds that callback in memory.
    A restart drops it, so nothing is left that could ever start the job.
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

    assert reconcile_interrupted_jobs(db) == 2

    for job in (running, pending, completed):
        db.refresh(job)
    assert running.status == "failed"
    assert running.error
    assert running.completed_at_utc is not None
    # Failed too, but with its own wording: it never ran at all, so
    # "interrupted before completion" would misdescribe it.
    assert pending.status == "failed"
    assert "Never started" in pending.error
    assert pending.completed_at_utc is not None
    assert completed.status == "completed"

    db.refresh(processing)
    db.refresh(pending_entry)
    assert processing.status == "failed"
    assert processing.error
    assert pending_entry.status == "pending"


def _run_pending_cleanup(task_id: str) -> None:
    """Register a pending start and immediately run its cleanup.

    `register_start` schedules the cleanup on the running loop, so this has
    to happen inside one. `_fail_orphaned_job` opens its own session via
    `get_session_factory`, which in tests would otherwise point at a
    different engine than the `db` fixture, so it is pointed at the shared
    in-memory one (same approach as the Camtrap worker helper).
    """
    import asyncio
    from unittest.mock import patch

    from app.core.websocket_manager import ws_manager
    from tests.conftest import _engine  # noqa: PLC2701 — shared test engine

    session_factory = sessionmaker(bind=_engine)

    async def _go() -> None:
        ws_manager.register_start(task_id, lambda: None)
        with patch("app.db.base.get_session_factory", lambda: session_factory):
            await ws_manager._cleanup_pending_start(task_id, delay=0)

    asyncio.run(_go())


def test_orphaned_pending_start_fails_the_job(db):
    """The 5-minute cleanup used to drop only the in-memory callback, so a
    job whose frontend never connected stayed `pending` for ever: nothing
    could start it, and startup reconciliation only looked at `running`.
    Now the cleanup settles the row too."""
    job = make_job(db, status="pending")
    db.commit()

    _run_pending_cleanup(job.id)

    db.expire_all()
    refreshed = db.get(Job, job.id)
    assert refreshed.status == "failed"
    assert "never connected" in refreshed.error


def test_pending_start_cleanup_ignores_a_task_that_is_not_a_job(db):
    """`register_start` is also used for model preparation, where the task
    id is a model id and no job row exists. The cleanup must not care."""
    _run_pending_cleanup("SPECIESNET-v4-0-2-A")  # must not raise


def test_pending_start_cleanup_leaves_a_job_that_ran(db):
    """A job that already started and finished must not be reopened by a
    late cleanup tick."""
    job = make_job(db, status="completed")
    db.commit()

    _run_pending_cleanup(job.id)

    db.expire_all()
    refreshed = db.get(Job, job.id)
    assert refreshed.status == "completed"
