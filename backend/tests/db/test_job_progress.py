"""
A progress write never fails an analysis run.

The job row carries the run's overall percent as a copy for readers
without the socket. It used to be written through the worker's own
session while the database load held SQLite's one write lock in another
thread. The write waited out the busy timeout, failed, and left the
worker session unusable, so a 191k-image run failed at the finish
(2026-11-22). Now `_write_job_percent` writes in a session of its own and
a locked database costs one tick.

These tests need a real SQLite file: an in-memory database has no second
connection that could hold the lock.
"""

import logging
import sqlite3

import pytest
from sqlalchemy import update

from app.db import base
from app.db.base import get_session_factory, init_db
from app.models import Job
from app.workers.detection_worker import _write_job_percent


@pytest.fixture()
def job_id(isolated_db_settings, monkeypatch) -> str:
    """A running job at 10% in a fresh database with a short busy timeout."""
    # Before the engine is built, so its connections wait 0.2 s, not 30 s.
    monkeypatch.setattr(base, "BUSY_TIMEOUT_S", 0.2)
    init_db()
    with get_session_factory()() as db:
        job = Job(type="deployment_analysis", status="running", progress_current=10)
        db.add(job)
        db.commit()
        return job.id


def _percent(job_id: str) -> int:
    with get_session_factory()() as db:
        return db.get(Job, job_id).progress_current


def test_a_locked_database_costs_a_tick_not_the_run(
    job_id, isolated_db_settings, caplog
):
    worker = get_session_factory()()
    worker.get(Job, job_id)  # the worker session is mid-run

    holder = sqlite3.connect(
        isolated_db_settings.user_data_dir / "addaxai.db", isolation_level=None
    )
    holder.execute("BEGIN IMMEDIATE")  # what the database load does for minutes
    try:
        with caplog.at_level(logging.WARNING):
            _write_job_percent(job_id, 11)  # must not raise
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    assert "progress 11% not written" in caplog.text
    assert _percent(job_id) == 10

    # The worker session carries on as if nothing happened. Before the fix
    # this raised PendingRollbackError.
    worker.execute(update(Job).where(Job.id == job_id).values(status="completed"))
    worker.commit()
    worker.close()

    # And the next tick lands.
    _write_job_percent(job_id, 12)
    assert _percent(job_id) == 12


def test_the_percent_only_goes_up(job_id):
    """Two writes waiting on the same lock can land in either order."""
    _write_job_percent(job_id, 96)
    _write_job_percent(job_id, 95)
    assert _percent(job_id) == 96
