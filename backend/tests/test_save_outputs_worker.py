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

from PIL import Image

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


def _seed_video_run(db, tmp_path: Path) -> tuple[str, Path, Path]:
    """A folder-run project with one video on disk and a real JPEG best
    frame holding a person box, so boxes and blur both have work to do.
    Returns (project_id, output_dir, container path)."""
    project = make_project(db, mode="folder_run")
    src_dir = tmp_path / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(src_dir)
    )
    clip = src_dir / "CLIP.mp4"
    clip.write_bytes(b"container-bytes")
    frame = tmp_path / "cache" / "frame000000.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 200), (180, 200, 220)).save(frame, format="JPEG")
    file = make_file(
        db,
        deployment_id=dep.id,
        file_path=str(clip),
        file_type="video",
        file_format="mp4",
        observation_type="person",
        best_frame_number=0,
        best_frame_path=str(frame),
    )
    make_detection(
        db, file_id=file.id, category="person", confidence=0.9, frame_number=0
    )
    db.commit()
    return project.id, tmp_path / "out", clip


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


def test_video_is_copied_whole_with_a_boxed_still_beside_it(
    db, tmp_path, monkeypatch
):
    """End to end through the worker: separation copies the container
    (even though its writes are deferred to annotation), and annotation
    puts the boxed best frame beside it as ``<stem>_still.jpg``."""
    project_id, out, clip = _seed_video_run(db, tmp_path)

    job = _run_save(db, monkeypatch, project_id, out, draw_bboxes=True)

    assert job.status == "completed"
    person = out / MEDIA_SUBDIR / "person"
    assert (person / "CLIP.mp4").read_bytes() == clip.read_bytes()
    assert (person / "CLIP_still.jpg").is_file()
    assert job.result["separate_folders"]["copied_count"] == 1
    assert job.result["annotated_copies"]["written_count"] == 1


def test_blur_writes_a_video_as_a_blurred_still_only(
    db, tmp_path, monkeypatch
):
    """The one exception to copying the clip: with blur on, the still is
    the only thing written, or the unblurred video would sit beside the
    blurred frame it came from."""
    project_id, out, _clip = _seed_video_run(db, tmp_path)

    job = _run_save(db, monkeypatch, project_id, out, anonymise=True)

    assert job.status == "completed"
    person = out / MEDIA_SUBDIR / "person"
    assert (person / "CLIP_still.jpg").is_file()
    assert not (person / "CLIP.mp4").exists()
    assert job.result["annotated_copies"]["blurred_box_count"] == 1


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


def _seed_two_camera_project(db, tmp_path: Path):
    """A research project with two deployments of one person photo each,
    the cam01 one liked. Returns (project_id, cam01 id, output_dir)."""
    project = make_project(db, mode="research")
    deps = {}
    for name, liked in (("cam01", True), ("cam02", False)):
        folder = tmp_path / "drive" / name
        folder.mkdir(parents=True)
        dep = make_deployment(db, project_id=project.id, folder_path=str(folder))
        src = folder / "IMG_000.jpg"
        src.write_bytes(b"jpegbytes")
        file = make_file(
            db,
            deployment_id=dep.id,
            file_path=str(src),
            observation_type="person",
            favorited=liked,
        )
        make_detection(db, file_id=file.id, category="person", confidence=0.9)
        deps[name] = dep
    db.commit()
    return project.id, deps["cam01"].id, tmp_path / "out"


def test_media_export_job_copies_only_the_scope_and_no_data_files(
    db, tmp_path, monkeypatch
):
    """The project image copies export is this worker with `project_id`,
    a scope and only the media modules. The deployment folder's name is
    kept in the layout, no readme or table lands in the user's folder,
    and the marker is stamped so a re-export can replace the tree."""
    project_id, cam01_id, out = _seed_two_camera_project(db, tmp_path)
    import app.workers.folder_run_save_outputs_worker as worker

    def _test_get_db():
        yield db

    monkeypatch.setattr(worker, "get_db", _test_get_db)
    job = make_job(
        db,
        job_type="media_export",
        payload={
            "project_id": project_id,
            "deployment_ids": [cam01_id],
            "favorited_only": True,
            "output_dir": str(out),
            "separate_folders": True,
            "run_readme": False,
            "media_threshold": 0.5,
        },
    )
    db.commit()
    asyncio.run(worker.process_save_outputs_job(job.id))
    db.expire_all()
    job = db.get(type(job), job.id)

    assert job.status == "completed", job.error
    media = out / MEDIA_SUBDIR
    copies = sorted(p.relative_to(media).as_posix() for p in media.rglob("*.jpg"))
    assert copies == ["person/cam01/IMG_000.jpg"]
    assert (media / OUTPUT_DIR_MARKER).is_file()
    assert job.result["source_file_count"] == 1
    assert [p.name for p in out.iterdir()] == [MEDIA_SUBDIR]


def test_folder_run_save_honours_the_shared_file_filters(
    db, tmp_path, monkeypatch
):
    """The liked / verified / date filters are plain media options, so a
    folder run sends them like a project does and the worker's scope
    reads them off a `run_id` payload. One of two files is verified: one
    copy, in the folder-run layout (no deployment folder segment)."""
    from app.models import File

    project_id, out = _seed_run(db, tmp_path, n_files=2)
    first = db.query(File).order_by(File.file_path).first()
    first.verified = True
    db.commit()

    job = _run_save(db, monkeypatch, project_id, out, verified_only=True)

    assert job.status == "completed", job.error
    media = out / MEDIA_SUBDIR
    copies = sorted(p.relative_to(media).as_posix() for p in media.rglob("*.jpg"))
    assert copies == ["person/IMG_000.jpg"]
    assert job.result["source_file_count"] == 1


def test_cancel_lands_inside_a_module_not_only_between_them(
    db, tmp_path, monkeypatch
):
    """A project-sized copy runs for hours, so the per-file progress
    callback checks for cancel too. Flipping the flag after the first
    file's callback stops the module with one copy written."""
    project_id, out = _seed_run(db, tmp_path, n_files=3)
    import app.workers.folder_run_save_outputs_worker as worker

    calls = {"n": 0}

    def _flip_after_first_file(job_id: str) -> bool:
        # Call 1: the between-module check. Call 2: the callback before
        # file 0. Call 3: the callback before file 1 -> cancel.
        calls["n"] += 1
        return calls["n"] >= 3

    monkeypatch.setattr(worker, "is_cancel_requested", _flip_after_first_file)
    job = _run_save(db, monkeypatch, project_id, out)

    assert job.status == "cancelled"
    copies = list((out / MEDIA_SUBDIR).rglob("IMG_*.jpg"))
    assert len(copies) == 1
