"""The stored media path form survives a deployment folder that resolves
elsewhere (mapped drive on Windows, symlink here).

Two things are pinned through a symlinked deployment folder:

1. `load_json_to_database` neither crashes on the video's `best_frame_path`
   nor stores the resolved form: every `File.file_path` starts with the
   folder the user picked.
2. `update_database_from_smoothed_results` builds its lookup key the same
   way, so every detection is found again. The two used to be kept equal
   only by both calling `resolve()`; now both call neither.

See `tests/ml/test_media_paths_never_resolved.py` for the why.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from app.ml.json_pipeline import load_json_to_database
from app.ml.postprocessing import update_database_from_smoothed_results
from app.models import Detection, File
from tests.integration.conftest import build_detection_json, write_json

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink creation needs privileges on Windows"
)

_CLASSES = {"1": "lion", "2": "zebra"}


def _images(classifications: list) -> list[dict]:
    return [
        {
            "file": "subdir/img_001.jpg",
            "detections": [
                {
                    "category": "1",
                    "conf": 0.9,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "classifications": classifications,
                }
            ],
        },
        {
            "file": "videos/clip.mp4",
            "best_frame_number": 3,
            "detections": [
                {
                    "category": "1",
                    "conf": 0.8,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "frame_number": 3,
                    "classifications": classifications,
                }
            ],
        },
    ]


def test_loading_through_a_symlink_stores_the_picked_form(deployment_scaffold):
    s = deployment_scaffold
    link = s["tmp_path"] / "link"
    link.symlink_to(s["deploy_dir"], target_is_directory=True)

    json_path = write_json(
        s["artifacts"] / "results.json",
        build_detection_json(_images([[1, 0.7], [2, 0.3]]), classification_categories=_CLASSES),
    )
    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=link,
            job_id=s["job"].id,
            db=s["db"],
            artifacts_folder=s["artifacts"],
        )

    files = s["db"].query(File).filter(File.deployment_id == s["deployment"].id).all()
    assert len(files) == 2
    assert all(f.file_path.startswith(str(link)) for f in files)

    video = next(f for f in files if f.file_type == "video")
    assert video.best_frame_path == str(
        s["artifacts"] / "video_frames" / "videos" / "clip.mp4" / "frame000003.jpg"
    )

    # Postprocessing keys its lookup the same way, so a relabel lands.
    counts = update_database_from_smoothed_results(
        deployment_id=s["deployment"].id,
        smoothed_results=build_detection_json(
            _images([[2, 0.8], [1, 0.2]]), classification_categories=_CLASSES
        ),
        deployment_folder=link,
        db=s["db"],
    )
    assert counts["errors"] == 0
    assert counts["updated"] == 2
    assert {d.label for d in s["db"].query(Detection).all()} == {"zebra"}
