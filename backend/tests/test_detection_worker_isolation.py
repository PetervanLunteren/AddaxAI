"""One failing deployment must not take the rest of the queue with it.

`_process_batch_job` used to wrap its whole loop over queue entries in a
single `try`. Any exception from any phase of any folder aborted the run,
marked every still-`processing` entry failed, and the folders after it
were never looked at. A forum report ("it writes to the error log and
interrupts all processing") is what surfaced it.

The queue row already carries a per-entry status and error, and
`RunQueueModal` already renders "N of M deployments" plus one log row per
failure, so the fix only had to make the worker produce that state.

**Fault injection without test hooks in production code.** A folder
holding one image plus a regular *file* named `.addaxai` makes
`mkdir_hidden_addaxai` raise `NotADirectoryError`, because the artifacts
directory cannot be created underneath a file. That lands inside the loop
just after the placeholder deployment is created, so it exercises the
rollback too, and it never reaches a model. Folders with no media take the
existing empty-folder path and also never reach a model. So the whole file
runs with no weights, no environments and no subprocesses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.job_cancellation import JobCancelledError
from app.models import Deployment, DeploymentQueue
from app.workers import detection_worker
from app.workers.detection_worker import _process_batch_job
from tests.conftest import make_job, make_project


class _FakeManifest:
    env = "addaxai-base"
    full_image_cls = False


class _FakeManifestManager:
    def get_model(self, model_id):
        return _FakeManifest()


class _FakeEnvManager:
    def get_python(self, env_name):
        return Path("/usr/bin/python3")


class _FakeModelStorage:
    def get_model_file(self, manifest):
        return Path("/nonexistent/md.pt")

    def get_model_path(self, manifest):
        return Path("/nonexistent")


class _FakeDetector:
    def __init__(self, *a, **kw):
        pass


@pytest.fixture(autouse=True)
def _no_real_models(monkeypatch):
    """Stand in for everything loaded before the loop.

    All four are module-level imports in `detection_worker`, and all four
    are constructed once up front rather than per entry. That is exactly
    what makes per-entry isolation safe, so faking them here costs nothing
    in fidelity and keeps the tests runnable on CI, which has no weights.
    """
    monkeypatch.setattr(
        detection_worker, "ManifestManager", _FakeManifestManager
    )
    monkeypatch.setattr(detection_worker, "EnvironmentManager", _FakeEnvManager)
    monkeypatch.setattr(detection_worker, "ModelStorage", _FakeModelStorage)
    monkeypatch.setattr(detection_worker, "MegaDetectorV1000", _FakeDetector)


def _empty_folder(tmp_path: Path, name: str) -> Path:
    """A folder the worker completes without touching a model."""
    folder = tmp_path / name
    folder.mkdir()
    return folder


def _broken_folder(tmp_path: Path, name: str) -> Path:
    """A folder that raises once the worker tries to make its artifacts dir."""
    folder = tmp_path / name
    folder.mkdir()
    (folder / "IMG_0001.jpg").write_bytes(b"not really a jpeg")
    (folder / ".addaxai").write_text("a file where a directory is expected")
    return folder


def _queue(db, project_id: str, folder: Path) -> DeploymentQueue:
    """A queue entry in the state the router leaves it in before a run."""
    entry = DeploymentQueue(
        project_id=project_id,
        folder_path=str(folder),
        status="processing",
        image_count=0,
        video_count=0,
    )
    db.add(entry)
    db.flush()
    return entry


def _seed(db, tmp_path, folders):
    """Project + job + one queue entry per folder. Returns (job, entries)."""
    project = make_project(
        db,
        # No classifier, so no per-crop phase is configured at all. Nothing
        # here reaches one anyway: the broken folder dies while making its
        # artifacts directory, which is before phase 1, and the empty
        # folders `continue` before any phase starts.
        #
        # `embedding_model_id` is deliberately left alone. Setting it to
        # None does not stick — the column carries a Python-side default of
        # "DINOV2-VITS14", which fires whenever the attribute is None at
        # insert — so writing None here would read as a guarantee the test
        # does not actually have.
        classification_model_id=None,
    )
    job = make_job(db)
    entries = [_queue(db, project.id, f) for f in folders]
    db.commit()
    return project, job, entries


async def test_run_totals_exclude_a_failed_deployment(db, tmp_path, monkeypatch):
    """A failed deployment must not be counted in the run's file total.

    The rollback takes its rows back out of the database, so counting the
    files it was going to have leaves the completion log and payload
    claiming media that is not there. The broken folder here holds one
    image and the two others hold none, so the total is 1 if the failed
    entry is counted and 0 if it is not.
    """
    captured: dict = {}

    async def fake_send_complete(job_id, success, message, data=None):
        captured["data"] = data or {}
        captured["message"] = message

    monkeypatch.setattr(
        detection_worker.ws_manager, "send_complete", fake_send_complete
    )

    project, job, entries = _seed(
        db,
        tmp_path,
        [
            _empty_folder(tmp_path, "one"),
            _broken_folder(tmp_path, "two"),
            _empty_folder(tmp_path, "three"),
        ],
    )

    await _process_batch_job(
        job.id, project.id, [e.id for e in entries], db
    )

    assert db.get(DeploymentQueue, entries[1].id).status == "failed"
    assert captured["data"]["total_files"] == 0
    assert captured["data"]["total_detections"] == 0
    # And the headline number is the successes, not everything attempted.
    assert captured["data"]["deployments_processed"] == 2


async def test_one_failed_deployment_does_not_stop_the_others(db, tmp_path):
    """The whole point: entry 2 fails, entries 1 and 3 still run."""
    project, job, entries = _seed(
        db,
        tmp_path,
        [
            _empty_folder(tmp_path, "one"),
            _broken_folder(tmp_path, "two"),
            _empty_folder(tmp_path, "three"),
        ],
    )

    await _process_batch_job(
        job.id, project.id, [e.id for e in entries], db
    )

    statuses = [db.get(DeploymentQueue, e.id).status for e in entries]
    assert statuses == ["completed", "failed", "completed"]

    failed = db.get(DeploymentQueue, entries[1].id)
    assert failed.error, "a failed entry with no error renders no log row"
    assert "Not a directory" in failed.error or "addaxai" in failed.error

    # A run where something landed is a completed run, which is what opens
    # the modal's summary block (it is gated on isComplete && !hasError).
    assert db.get(type(job), job.id).status == "completed"


async def test_failed_entry_leaves_no_placeholder_deployment(db, tmp_path):
    """The placeholder row is created before the failing step.

    Without the rollback in the handler it survives as an orphan on the
    Deployments page: today's date, the failed folder, zero files.
    """
    project, job, entries = _seed(
        db, tmp_path, [_broken_folder(tmp_path, "broken"), _empty_folder(tmp_path, "ok")]
    )

    await _process_batch_job(
        job.id, project.id, [e.id for e in entries], db
    )

    assert db.query(Deployment).count() == 0


async def test_all_failed_marks_the_job_failed(db, tmp_path):
    """Nothing landed, so the run is a failure, exactly as before.

    This is what keeps a one-folder run honest. Every folder run enqueues
    exactly one entry, so without this rule a folder run whose only
    deployment failed would render "Analysis complete / Processed the
    folder" above its own error.
    """
    project, job, entries = _seed(
        db, tmp_path, [_broken_folder(tmp_path, "only")]
    )

    await _process_batch_job(
        job.id, project.id, [e.id for e in entries], db
    )

    assert db.get(DeploymentQueue, entries[0].id).status == "failed"
    assert db.get(type(job), job.id).status == "failed"


async def test_cancel_still_aborts_the_whole_queue(db, tmp_path, monkeypatch):
    """A cancel is not a deployment failure.

    `except JobCancelledError: raise` has to sit in front of the broad
    handler. Behind it, a cancel would be recorded as one folder failing
    and the queue would carry on through every remaining folder, ignoring
    the button the user just pressed.
    """
    project, job, entries = _seed(
        db,
        tmp_path,
        [
            _empty_folder(tmp_path, "done"),
            _broken_folder(tmp_path, "cancel-here"),
            _empty_folder(tmp_path, "never-reached"),
        ],
    )

    def _cancel(*a, **kw):
        raise JobCancelledError()

    monkeypatch.setattr(detection_worker, "mkdir_hidden_addaxai", _cancel)

    await _process_batch_job(
        job.id, project.id, [e.id for e in entries], db
    )

    statuses = [db.get(DeploymentQueue, e.id).status for e in entries]
    # Entry 1 finished before the cancel and keeps its result. The
    # in-flight entry and everything after it go back to pending so the
    # user can re-run them without re-adding the folders.
    assert statuses == ["completed", "pending", "pending"]
    assert db.get(type(job), job.id).status == "cancelled"
