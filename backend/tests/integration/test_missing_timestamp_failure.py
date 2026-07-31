"""
Integration tests: capture-timestamp handling during JSON load.

Folder mode is data-agnostic: a file with no resolvable capture date is
still ingested, with `captured_at_local = NULL`, and surfaced via
`PipelineResult.skipped_missing_timestamp` so the UI can report how many
lack a date. Time-based stats exclude the null-date rows and each
becomes its own single-file event downstream. Nothing is dropped, and
there is no hard failure even when every file lacks a timestamp.

Observational datetimes are never guessed. The one exception is the
per-folder `use_file_mtime_fallback` opt-in, which is off in every test
here, so these also pin the default behaviour: an mtime exists for all of
these files and is deliberately not used. See DEVELOPERS.md "Datetime
conventions", and tests/integration/test_mtime_fallback.py for the
opted-in path.
"""

from unittest.mock import patch

from app.ml.json_pipeline import load_json_to_database
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


def test_missing_timestamp_partial_ingested_with_null(deployment_scaffold):
    """
    One file has a valid DateTimeOriginal, one does not. Both are
    ingested: the dated file keeps its timestamp, the undated file gets
    captured_at_local=NULL and is still reported in
    `skipped_missing_timestamp`. No exception, deployment kept.
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

    # Both files loaded (data-agnostic), both detections kept.
    files = (
        db.query(File).filter(File.deployment_id == s["deployment"].id).all()
    )
    assert len(files) == 2
    by_path = {f.file_path: f for f in files}
    assert by_path[str(s["img_paths"][0])].captured_at_local is not None
    assert by_path[str(s["img_paths"][1])].captured_at_local is None

    det_count = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .filter(File.deployment_id == s["deployment"].id)
        .count()
    )
    assert det_count == 2

    # Undated file still surfaced in the result for the UI count.
    assert len(result.skipped_missing_timestamp) == 1
    assert bad_rel in result.skipped_missing_timestamp[0]

    # Deployment dates derive from the one dated file (min/max ignore NULL).
    dep = db.query(Deployment).filter(Deployment.id == s["deployment"].id).one()
    assert dep.start_date_local.isoformat() == "2024-06-15"
    assert dep.end_date_local.isoformat() == "2024-06-15"


def test_all_files_missing_timestamps_ingested(deployment_scaffold):
    """
    Zero files have a resolvable timestamp. They are all ingested with
    captured_at_local=NULL — no hard failure, deployment kept, every file
    surfaced in `skipped_missing_timestamp`.
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
        result = load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=deploy_dir,
            job_id=s["job"].id,
            db=db,
            artifacts_folder=s["artifacts"],
        )

    files = (
        db.query(File).filter(File.deployment_id == s["deployment"].id).all()
    )
    assert len(files) == len(s["img_paths"])
    assert all(f.captured_at_local is None for f in files)
    assert len(result.skipped_missing_timestamp) == len(s["img_paths"])


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
