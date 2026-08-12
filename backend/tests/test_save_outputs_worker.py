"""End-to-end tests for the folder-run save-outputs worker.

Cover the two behaviors added after Wayne's 2026-08 report:
- retries rebuild the media tree instead of duplicating every copy
  with `_2` / `_3` suffixes (nothing covered a second save into a
  non-empty ``addaxai-media`` before)
- per-module start/done log lines with elapsed time and process RSS,
  the only trace left when a module kills the process.
"""

import asyncio
import logging
from pathlib import Path

from app.ml.postprocessing_outputs._output_context import MEDIA_SUBDIR
from app.services.folder_scanner import OUTPUT_DIR_MARKER
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_job,
    make_project,
)


def _seed_run(db, tmp_path: Path, n_files: int = 2) -> tuple[str, Path]:
    """A folder-run project with person images on disk; returns
    (project_id, output_dir)."""
    project = make_project(db, mode="folder_run")
    src_dir = tmp_path / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(src_dir)
    )
    for i in range(n_files):
        src = src_dir / f"IMG_{i:03d}.jpg"
        src.write_bytes(b"jpegbytes")
        file = make_file(
            db,
            deployment_id=dep.id,
            file_path=str(src),
            observation_type="person",
        )
        make_detection(
            db, file_id=file.id, category="person", confidence=0.9
        )
    db.commit()
    return project.id, tmp_path / "out"


def _run_save(db, monkeypatch, project_id: str, output_dir: Path, **payload):
    """Create a save job and run the worker on the test DB."""
    import app.workers.folder_run_save_outputs_worker as worker

    def _test_get_db():
        yield db

    monkeypatch.setattr(worker, "get_db", _test_get_db)

    job = make_job(
        db,
        job_type="folder_run_save_outputs",
        payload={
            "run_id": project_id,
            "output_dir": str(output_dir),
            "separate_folders": True,
            "recognition_json": False,
            "csv": False,
            "xlsx": False,
            "run_readme": False,
            "media_threshold": 0.5,
            **payload,
        },
    )
    db.commit()
    asyncio.run(worker.process_save_outputs_job(job.id))
    db.expire_all()
    return db.get(type(job), job.id)


def test_second_save_replaces_media_copies(db, tmp_path, monkeypatch):
    project_id, out = _seed_run(db, tmp_path)

    first = _run_save(db, monkeypatch, project_id, out)
    assert first.status == "completed"
    media = out / MEDIA_SUBDIR
    copies = sorted(p.name for p in media.rglob("IMG_*.jpg"))
    assert copies == ["IMG_000.jpg", "IMG_001.jpg"]

    second = _run_save(db, monkeypatch, project_id, out)
    assert second.status == "completed"
    copies_after = sorted(p.name for p in media.rglob("IMG_*.jpg"))
    assert copies_after == ["IMG_000.jpg", "IMG_001.jpg"]
    assert second.result["separate_folders"]["renamed_count"] == 0


def test_markerless_media_dir_is_not_wiped_or_claimed(
    db, tmp_path, monkeypatch
):
    """A pre-existing addaxai-media without our marker is not ours to
    delete; its content survives the save. It must not be stamped
    either: a stamp would hand the NEXT save's wipe the ownership proof
    and delete the same files one save later."""
    from app.services.folder_scanner import OUTPUT_DIR_MARKER

    project_id, out = _seed_run(db, tmp_path)
    media = out / MEDIA_SUBDIR
    media.mkdir(parents=True)
    foreign = media / "users-own-file.txt"
    foreign.write_text("keep me")

    first = _run_save(db, monkeypatch, project_id, out)
    assert first.status == "completed"
    assert foreign.read_text() == "keep me"
    assert not (media / OUTPUT_DIR_MARKER).exists()

    # The property must hold on every later save, not only the first.
    second = _run_save(db, monkeypatch, project_id, out)
    assert second.status == "completed"
    assert foreign.read_text() == "keep me"
    assert not (media / OUTPUT_DIR_MARKER).exists()


def test_data_only_save_never_wipes_media(db, tmp_path, monkeypatch):
    """Without media modules there is nothing to rebuild: an existing
    (marker-stamped) media tree from an earlier save stays untouched."""
    project_id, out = _seed_run(db, tmp_path)
    media = out / MEDIA_SUBDIR
    media.mkdir(parents=True)
    (media / OUTPUT_DIR_MARKER).touch()
    earlier = media / "person" / "IMG_000.jpg"
    earlier.parent.mkdir(parents=True)
    earlier.write_bytes(b"from an earlier save")

    job = _run_save(
        db,
        monkeypatch,
        project_id,
        out,
        separate_folders=False,
        run_readme=True,
    )
    assert job.status == "completed"
    assert earlier.read_bytes() == b"from an earlier save"


def test_failed_save_persists_error_on_job(db, tmp_path, monkeypatch):
    """The failure message must land on the job row, not only on the
    live WebSocket: after a reload or restart it is all the user (and a
    diagnostics bundle) has. Wayne's save jobs all read just
    'Interrupted by a server restart before completion'."""
    import app.workers.folder_run_save_outputs_worker as worker

    def _test_get_db():
        yield db

    monkeypatch.setattr(worker, "get_db", _test_get_db)
    job = make_job(
        db,
        job_type="folder_run_save_outputs",
        payload={"run_id": "whatever"},  # no output_dir -> ValueError
    )
    db.commit()
    asyncio.run(worker.process_save_outputs_job(job.id))
    db.expire_all()

    refreshed = db.get(type(job), job.id)
    assert refreshed.status == "failed"
    assert "output_dir" in (refreshed.error or "")


def test_module_log_lines_carry_elapsed_and_rss(
    db, tmp_path, monkeypatch, caplog
):
    project_id, out = _seed_run(db, tmp_path)

    with caplog.at_level(logging.INFO):
        job = _run_save(db, monkeypatch, project_id, out)

    assert job.status == "completed"
    starts = [
        r.message
        for r in caplog.records
        if r.message.startswith("save_outputs: module=separate_folders start")
    ]
    dones = [
        r.message
        for r in caplog.records
        if r.message.startswith("save_outputs: module=separate_folders done")
    ]
    assert len(starts) == 1 and "rss_mb=" in starts[0]
    assert len(dones) == 1
    assert "elapsed_s=" in dones[0] and "rss_mb=" in dones[0]
