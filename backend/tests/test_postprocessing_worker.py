"""Tests for the reprocess (postprocessing) worker.

A reprocess re-reads each deployment's results.json and re-applies the
retroactive settings. A deployment whose results.json was moved or
deleted cannot be re-applied, and the worker used to skip it in silence:
the job reported success and stamped the project's settings hash, so
``/postprocessing-status`` then answered "no reprocess needed" for
labels that were still built with the old settings.

These pin the honest behavior: say how many were skipped, and leave the
hash alone until every deployment really has been reprocessed.
"""

import asyncio
import json
from pathlib import Path

from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_job,
    make_project,
)


def _seed(db, tmp_path: Path, *, folder_path: str | None = "") -> tuple[str, Path]:
    """A project with one classified deployment. Returns (project_id, folder).

    ``folder_path=None`` makes the deployment carry no folder at all, which
    the API allows and which used to crash the worker.
    """
    project = make_project(db, taxonomic_rollup=True, event_smoothing=False)
    folder = tmp_path / "deployment-01"
    folder.mkdir(parents=True, exist_ok=True)
    dep = make_deployment(
        db,
        project_id=project.id,
        folder_path=str(folder) if folder_path == "" else folder_path,
    )
    image = folder / "IMG_000.jpg"
    image.write_bytes(b"jpegbytes")
    file = make_file(
        db, deployment_id=dep.id, file_path=str(image.resolve())
    )
    make_detection(db, file_id=file.id, label="fox", label_confidence=0.9)
    db.commit()
    return project.id, folder


def _write_results_json(folder: Path, project_id: str) -> None:
    """Minimal results.json matching the seeded file and bbox."""
    path = folder / ".addaxai" / "projects" / project_id / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "images": [{
                "file": "IMG_000.jpg",
                "detections": [{
                    "category": "1",
                    "conf": 0.9,
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                    "classifications": [["1", 0.88]],
                }],
            }],
            "detection_categories": {"1": "animal"},
            "classification_categories": {"1": "badger"},
        })
    )


def _run_job(db, monkeypatch, project_id: str) -> dict:
    """Run the worker on the test DB; returns the completion payload."""
    import app.workers.postprocessing_worker as worker

    def _test_get_db():
        yield db

    monkeypatch.setattr(worker, "get_db", _test_get_db)

    sent: dict = {}

    class _Recorder:
        async def send_progress(self, *a, **kw):
            pass

        async def send_error(self, *a, **kw):
            sent["error"] = a

        async def send_complete(self, **kw):
            sent.update(kw)

    monkeypatch.setattr(worker, "ws_manager", _Recorder())

    job = make_job(
        db, job_type="postprocessing", payload={"project_id": project_id}
    )
    db.commit()
    asyncio.run(worker.process_postprocessing_job(job.id))
    db.expire_all()
    return sent


def test_missing_results_json_is_reported_and_leaves_hash_unset(
    db, tmp_path, monkeypatch
):
    """Folder is there, its .addaxai artifacts are not: reported under
    `no_results`, and the project still reads as needing a reprocess."""
    from app.models import Project

    project_id, folder = _seed(db, tmp_path)

    sent = _run_job(db, monkeypatch, project_id)

    assert sent["success"] is True
    assert sent["data"]["skipped"] == {
        "no_results": {"count": 1, "path": str(folder)}
    }
    assert sent["message"] == "Could not apply settings to any folder"
    assert db.get(Project, project_id).postprocessing_settings_hash is None


def test_missing_folder_is_reported_as_a_different_cause(
    db, tmp_path, monkeypatch
):
    """Folder gone entirely: a reconnect fixes that, not a new analysis,
    so it is reported apart from a folder that only lost its results."""
    project_id, folder = _seed(db, tmp_path)
    (folder / "IMG_000.jpg").unlink()
    folder.rmdir()

    sent = _run_job(db, monkeypatch, project_id)

    assert sent["data"]["skipped"] == {
        "folder_missing": {"count": 1, "path": str(folder)}
    }


def test_reprocessed_deployment_stamps_hash_and_reports_no_skips(
    db, tmp_path, monkeypatch
):
    """The other half: with results.json present nothing is skipped and
    the hash is stamped, so the UI stops asking for a reprocess."""
    from app.models import Project

    project_id, folder = _seed(db, tmp_path)
    _write_results_json(folder, project_id)

    sent = _run_job(db, monkeypatch, project_id)

    assert sent["data"]["skipped"] == {}
    assert "skipped" not in sent["message"]
    assert sent["message"].startswith("Settings applied to 1 of 1 folders")
    assert db.get(Project, project_id).postprocessing_settings_hash is not None


def test_unreadable_results_are_skipped_not_fatal(db, tmp_path, monkeypatch):
    """A results.json that cannot be parsed is one skipped folder, not a
    silent success. It used to report `Smoothing applied` and stamp the
    hash while the settings never reached that folder."""
    from app.models import Project

    project_id, folder = _seed(db, tmp_path)
    _write_results_json(folder, project_id)
    (folder / ".addaxai" / "projects" / project_id / "results.json").write_text(
        "{ this is not json "
    )

    sent = _run_job(db, monkeypatch, project_id)

    assert sent["data"]["skipped"] == {
        "unreadable": {"count": 1, "path": str(folder)}
    }
    assert db.get(Project, project_id).postprocessing_settings_hash is None


def test_unreadable_folder_does_not_abort_the_whole_job(
    db, tmp_path, monkeypatch
):
    """Path.exists() re-raises EACCES. One locked folder used to end the
    job for every other deployment in the project, with the raw errno
    shown to the user."""
    project_id, folder = _seed(db, tmp_path)
    artifacts = folder / ".addaxai"
    artifacts.mkdir(parents=True, exist_ok=True)
    artifacts.chmod(0o000)
    try:
        sent = _run_job(db, monkeypatch, project_id)
    finally:
        artifacts.chmod(0o755)

    assert sent["success"] is True
    assert sent["data"]["skipped"] == {
        "unreadable": {"count": 1, "path": str(folder)}
    }


def test_deployment_without_a_folder_is_skipped_not_fatal(
    db, tmp_path, monkeypatch
):
    """folder_path is nullable and handled everywhere else; Path(None)
    here used to kill the job with a TypeError."""
    project_id, _folder = _seed(db, tmp_path, folder_path=None)

    sent = _run_job(db, monkeypatch, project_id)

    assert sent["success"] is True
    assert list(sent["data"]["skipped"]) == ["no_folder"]
    assert sent["data"]["skipped"]["no_folder"]["count"] == 1
