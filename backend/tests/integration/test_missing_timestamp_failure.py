"""
Integration tests: capture-timestamp handling during JSON load.

Partial failures (some files have a timestamp, some don't) are a
**soft** skip: the unresolvable rows are dropped, the list surfaces
via `PipelineResult.skipped_missing_timestamp`, and the rest of the
deployment loads normally. The all-broken case (zero resolvable
timestamps) is still a hard `MissingTimestampError` so the worker
rolls back the placeholder deployment instead of leaving a zombie
empty row in the DB.

Observational datetimes are never guessed — see DEVELOPERS.md
"Datetime conventions".
"""

from unittest.mock import patch

import pytest

from app.ml.json_pipeline import MissingTimestampError, load_json_to_database
from app.models import Deployment, Detection, File

from .conftest import build_detection_json, write_json


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


def test_missing_timestamp_partial_soft_skip(deployment_scaffold):
    """
    One file has a valid DateTimeOriginal, one does not. The loader must
    keep going, load the good file normally, report the bad file in
    `PipelineResult.skipped_missing_timestamp`, and leave the Deployment
    row in place. No exception.
    """
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    good_rel = str(s["img_paths"][0].relative_to(deploy_dir))
    bad_rel = str(s["img_paths"][1].relative_to(deploy_dir))

    images = [
        _image_entry(good_rel, {"DateTimeOriginal": "2024:06:15 12:00:00"}),
        _image_entry(bad_rel, {}),  # empty exif => no DateTimeOriginal
    ]
    md_json = build_detection_json(images)
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        result = load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    # Good file loaded, bad file skipped.
    file_count = (
        db.query(File).filter(File.deployment_id == s["deployment"].id).count()
    )
    assert file_count == 1

    det_count = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .filter(File.deployment_id == s["deployment"].id)
        .count()
    )
    assert det_count == 1

    # Skipped paths surfaced in the result.
    assert len(result.skipped_missing_timestamp) == 1
    assert bad_rel in result.skipped_missing_timestamp[0]

    # Deployment dates derived from the one good file's timestamp.
    dep = db.query(Deployment).filter(Deployment.id == s["deployment"].id).one()
    assert dep.start_date_local.isoformat() == "2024-06-15"
    assert dep.end_date_local.isoformat() == "2024-06-15"


def test_all_files_missing_timestamps_raises(deployment_scaffold):
    """
    Zero files have a resolvable timestamp. Nothing to ingest — the
    loader must raise `MissingTimestampError` so the worker can roll
    back the placeholder deployment. No partial DB state should survive.
    """
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    images = []
    for p in s["img_paths"]:
        rel = str(p.relative_to(deploy_dir))
        images.append(_image_entry(rel, {}))

    md_json = build_detection_json(images)
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        with pytest.raises(MissingTimestampError) as exc_info:
            load_json_to_database(
                json_path=json_path,
                deployment_id=s["deployment"].id,
                deployment_folder=deploy_dir,
                job_id=s["job"].id,
                db=db,
                artifacts_folder=s["artifacts"],
            )

    err = exc_info.value
    assert len(err.missing_paths) == 3
    assert "3 file(s)" in str(err)

    # No File / Detection rows were created (we raised before the main
    # insert loop had anything to persist).
    file_count = (
        db.query(File).filter(File.deployment_id == s["deployment"].id).count()
    )
    det_count = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .filter(File.deployment_id == s["deployment"].id)
        .count()
    )
    assert file_count == 0
    assert det_count == 0


def test_all_files_present_no_warning(deployment_scaffold):
    """Every file has a DateTimeOriginal → clean load, empty skip list."""
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    images = []
    for i, p in enumerate(s["img_paths"]):
        rel = str(p.relative_to(deploy_dir))
        images.append(
            _image_entry(
                rel,
                {"DateTimeOriginal": f"2024:06:{15 + i:02d} 12:00:00"},
            )
        )

    md_json = build_detection_json(images)
    json_path = write_json(s["artifacts"] / "results.json", md_json)

    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        result = load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    assert result.skipped_missing_timestamp == []
    file_count = (
        db.query(File).filter(File.deployment_id == s["deployment"].id).count()
    )
    assert file_count == len(s["img_paths"])
