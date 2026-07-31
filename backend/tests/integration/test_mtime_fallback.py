"""
Integration tests: the opt-in file-modification-time capture-date fallback.

A user whose folder carries no capture dates can tick a box in the folder
scan, after seeing the exact date range it would produce, and have the
ingest read each file's modification time instead. It is off by default
and it never overrides a real capture date.

The flag-off behaviour lives in test_missing_timestamp_failure.py.
"""

import os
from datetime import datetime
from unittest.mock import patch

from app.ml.json_pipeline import load_json_to_database
from app.models import Deployment, File

from .conftest import build_detection_json, write_json

# Distinct per-file mtimes so a test can tell which file a date came from.
MTIME_A = datetime(2024, 4, 7, 15, 55, 26)
MTIME_B = datetime(2024, 4, 9, 8, 30, 0)


def _image_entry(rel_path: str, exif: dict | None = None) -> dict:
    """Build a single images[] entry. exif=None means "no EXIF at all"."""
    entry: dict = {
        "file": rel_path,
        "detections": [
            {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4]},
        ],
    }
    if exif is not None:
        entry["exif_metadata"] = exif
    return entry


def _set_mtime(path, when: datetime) -> None:
    os.utime(path, (when.timestamp(), when.timestamp()))


def _load(scaffold, images, **kwargs):
    """Run the DB load over ``images`` with exiftool stubbed out."""
    json_path = write_json(
        scaffold["artifacts"] / "results.json", build_detection_json(images)
    )
    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        return load_json_to_database(
            json_path=json_path,
            deployment_id=scaffold["deployment"].id,
            deployment_folder=scaffold["deploy_dir"],
            job_id=scaffold["job"].id,
            db=scaffold["db"],
            artifacts_folder=scaffold["artifacts"],
            **kwargs,
        )


def test_fallback_off_ignores_mtime(deployment_scaffold):
    """The default. An mtime exists and is deliberately not used."""
    s = deployment_scaffold
    _set_mtime(s["img_paths"][0], MTIME_A)
    rel = str(s["img_paths"][0].relative_to(s["deploy_dir"]))

    result = _load(s, [_image_entry(rel, {})])

    file_row = s["db"].query(File).filter(File.file_path == str(s["img_paths"][0])).one()
    assert file_row.captured_at_local is None
    assert len(result.skipped_missing_timestamp) == 1


def test_fallback_on_uses_mtime(deployment_scaffold):
    """The undated file gets its modification time, and stops counting as
    dateless so the "N files have no capture date" notice is accurate."""
    s = deployment_scaffold
    _set_mtime(s["img_paths"][0], MTIME_A)
    rel = str(s["img_paths"][0].relative_to(s["deploy_dir"]))

    result = _load(s, [_image_entry(rel, {})], use_file_mtime_fallback=True)

    file_row = s["db"].query(File).filter(File.file_path == str(s["img_paths"][0])).one()
    assert file_row.captured_at_local == MTIME_A
    assert result.skipped_missing_timestamp == []


def test_fallback_on_derives_the_deployment_window(deployment_scaffold):
    """Without this the deployment keeps the creation-day placeholder the
    worker stamped, which is the visible symptom users report."""
    s = deployment_scaffold
    _set_mtime(s["img_paths"][0], MTIME_A)
    _set_mtime(s["img_paths"][1], MTIME_B)
    rels = [str(p.relative_to(s["deploy_dir"])) for p in s["img_paths"][:2]]

    _load(
        s,
        [_image_entry(rels[0], {}), _image_entry(rels[1], {})],
        use_file_mtime_fallback=True,
    )

    dep = s["db"].query(Deployment).filter(Deployment.id == s["deployment"].id).one()
    assert dep.start_date_local == MTIME_A.date()
    assert dep.end_date_local == MTIME_B.date()


def test_real_capture_date_always_wins(deployment_scaffold):
    """Gap-fill only. A file with real EXIF keeps it even though its mtime
    is readable and would parse fine."""
    s = deployment_scaffold
    _set_mtime(s["img_paths"][0], MTIME_A)
    _set_mtime(s["img_paths"][1], MTIME_B)
    dated_rel = str(s["img_paths"][0].relative_to(s["deploy_dir"]))
    undated_rel = str(s["img_paths"][1].relative_to(s["deploy_dir"]))

    _load(
        s,
        [
            _image_entry(dated_rel, {"DateTimeOriginal": "2024:06:15 12:00:00"}),
            _image_entry(undated_rel, {}),
        ],
        use_file_mtime_fallback=True,
    )

    rows = {
        f.file_path: f.captured_at_local
        for f in s["db"].query(File).filter(
            File.deployment_id == s["deployment"].id
        )
    }
    assert rows[str(s["img_paths"][0])] == datetime(2024, 6, 15, 12, 0, 0)
    assert rows[str(s["img_paths"][1])] == MTIME_B


def test_addaxai_filename_beats_mtime(deployment_scaffold):
    """Resolution order: the filename marker is a deliberate statement by
    the user about one file, mtime is a blanket fallback, so the marker
    wins. mtime must stay last because it succeeds for every readable
    file and would otherwise shadow everything below it."""
    s = deployment_scaffold
    marked = s["deploy_dir"] / "subdir" / "clip_addaxai-20250222-072314.jpg"
    s["img_paths"][0].rename(marked)
    _set_mtime(marked, MTIME_A)
    rel = str(marked.relative_to(s["deploy_dir"]))

    _load(s, [_image_entry(rel, {})], use_file_mtime_fallback=True)

    file_row = s["db"].query(File).filter(File.file_path == str(marked)).one()
    assert file_row.captured_at_local == datetime(2025, 2, 22, 7, 23, 14)


def test_video_falls_back_when_exiftool_finds_nothing(deployment_scaffold):
    """The AVI case that prompted the feature: no container date at all."""
    s = deployment_scaffold
    _set_mtime(s["vid_path"], MTIME_A)
    rel = str(s["vid_path"].relative_to(s["deploy_dir"]))

    _load(s, [_image_entry(rel, {})], use_file_mtime_fallback=True)

    file_row = s["db"].query(File).filter(File.file_path == str(s["vid_path"])).one()
    assert file_row.captured_at_local == MTIME_A


def test_video_metadata_beats_mtime(deployment_scaffold):
    """A video whose container does carry a date keeps it."""
    s = deployment_scaffold
    _set_mtime(s["vid_path"], MTIME_A)
    rel = str(s["vid_path"].relative_to(s["deploy_dir"]))
    real = datetime(2024, 6, 15, 12, 0, 0)

    json_path = write_json(
        s["artifacts"] / "results.json",
        build_detection_json([_image_entry(rel, {})]),
    )
    with patch(
        "app.ml.json_pipeline.extract_video_dates",
        return_value={s["vid_path"].resolve(): real},
    ):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=s["deploy_dir"],
            job_id=s["job"].id,
            db=s["db"],
            artifacts_folder=s["artifacts"],
            use_file_mtime_fallback=True,
        )

    file_row = s["db"].query(File).filter(File.file_path == str(s["vid_path"])).one()
    assert file_row.captured_at_local == real


def test_datetime_offset_applies_on_top_of_mtime(deployment_scaffold):
    """The offset is the documented remedy for a computer clock that ran
    on a different timezone than the camera, so it has to reach these."""
    s = deployment_scaffold
    _set_mtime(s["img_paths"][0], MTIME_A)
    rel = str(s["img_paths"][0].relative_to(s["deploy_dir"]))

    _load(
        s,
        [_image_entry(rel, {})],
        use_file_mtime_fallback=True,
        datetime_offset_seconds=-3600,
    )

    file_row = s["db"].query(File).filter(File.file_path == str(s["img_paths"][0])).one()
    assert file_row.captured_at_local == datetime(2024, 4, 7, 14, 55, 26)
