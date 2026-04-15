"""
Integration test: Phase 6 aborts with MissingTimestampError when any file
in the batch has no extractable capture timestamp.

Observational datetimes are never guessed — see DEVELOPERS.md
"Datetime conventions". This test guards the failure path end-to-end:
the bad file is reported, no DB rows are left behind, and other files in
the same batch don't leak through.
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


def test_missing_timestamp_raises_and_leaves_no_rows(deployment_scaffold):
    """
    One file with a valid DateTimeOriginal, one without. Phase 6 must
    raise MissingTimestampError, list the bad file, and roll back: no
    File / Detection / Deployment-date changes should persist.
    """
    s = deployment_scaffold
    db, deploy_dir = s["db"], s["deploy_dir"]

    good_rel = str(s["img_paths"][0].relative_to(deploy_dir))
    bad_rel = str(s["img_paths"][1].relative_to(deploy_dir))

    images = [
        _image_entry(good_rel, {"DateTimeOriginal": "2024:06:15 12:00:00"}),
        # Passing an empty dict so build_detection_json doesn't auto-fill.
        _image_entry(bad_rel, {}),
    ]
    md_json = build_detection_json(images)
    # The builder only defaults exif_metadata when DateTimeOriginal is
    # missing, but the copy we just wrote still has an empty dict — which
    # is the same as "no extractable timestamp". Assert that condition is
    # still present post-build so the test stays faithful to intent.
    assert "DateTimeOriginal" not in md_json["images"][1].get("exif_metadata", {})

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
    assert len(err.missing_paths) == 1
    assert bad_rel in err.missing_paths[0]
    assert "1 file(s)" in str(err)

    # Phase 6 pre-flights timestamps before touching the DB, so nothing
    # should have been written even for the "good" file.
    file_count = db.query(File).filter(File.deployment_id == s["deployment"].id).count()
    det_count = (
        db.query(Detection)
        .join(File, File.id == Detection.file_id)
        .filter(File.deployment_id == s["deployment"].id)
        .count()
    )
    assert file_count == 0
    assert det_count == 0
    # Deployment dates were untouched (Phase 6 derives them from
    # File.captured_at_local on success only).
    dep = db.query(Deployment).filter(Deployment.id == s["deployment"].id).one()
    assert dep.start_date_local.isoformat() == "2024-01-01"


def test_multiple_missing_files_all_reported(deployment_scaffold):
    """Every file missing a timestamp ends up in missing_paths; the first
    5 are surfaced in the exception message for log brevity."""
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
