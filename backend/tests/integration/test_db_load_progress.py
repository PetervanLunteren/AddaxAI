"""Progress and cancellation for the database load.

The load is the longest step of a run: a beta tester's million-image
ingest sat in it for over 12 hours showing a bar at 0%, with no way to
stop it. Both of those are what these tests pin.

The count is the fiddly part. The loader streams the results JSON with
ijson, so nothing knows how many entries there are until something walks
them, and MegaDetector failure entries (an undecodable video) are skipped
for insertion but still have to be counted, or the bar stops short of the
end on exactly the deployments most likely to worry someone.
"""

from unittest.mock import patch

import pytest

from app.core.job_cancellation import JobCancelledError
from app.ml.json_pipeline import load_json_to_database
from tests.integration.conftest import write_json


def _load(scaffold, images, **kwargs):
    """Run the loader over `images` with exiftool stubbed out."""
    s = scaffold
    from tests.integration.conftest import build_detection_json

    json_path = write_json(
        s["artifacts"] / "results.json", build_detection_json(images)
    )
    with patch("app.ml.json_pipeline.extract_video_dates", return_value={}):
        return load_json_to_database(
            json_path=json_path,
            deployment_id=s["deployment"].id,
            deployment_folder=s["deploy_dir"],
            job_id=s["job"].id,
            db=s["db"],
            artifacts_folder=s["artifacts"],
            **kwargs,
        )


def _image_entries(scaffold):
    return [
        {
            "file": str(p.relative_to(scaffold["deploy_dir"])),
            "detections": [
                {"category": "1", "conf": 0.8, "bbox": [0.1, 0.2, 0.3, 0.4]}
            ],
        }
        for p in scaffold["img_paths"]
    ]


def test_reports_progress_and_reaches_the_total(deployment_scaffold):
    """The bar must start with a total and end exactly on it."""
    calls: list[tuple[int, int]] = []
    entries = _image_entries(deployment_scaffold)

    _load(
        deployment_scaffold,
        entries,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert calls, "the loader reported no progress at all"
    # Emitted up front so the count is visible immediately.
    assert calls[0] == (0, len(entries))
    # And it lands on the total, not one short of it.
    assert calls[-1] == (len(entries), len(entries))
    assert all(total == len(entries) for _, total in calls)


def test_failed_video_entries_are_counted_in_the_total(deployment_scaffold):
    """
    A failure entry is skipped for insertion but still counted.

    Counting after the skip is the easy mistake, and it leaves the bar
    permanently short of the end on any deployment holding an undecodable
    video, which is precisely when a user is already suspicious.
    """
    calls: list[tuple[int, int]] = []
    entries = _image_entries(deployment_scaffold)
    entries.append({"file": "broken.mp4", "failure": "Video could not be read"})

    result = _load(
        deployment_scaffold,
        entries,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert calls[-1] == (len(entries), len(entries))
    # The failure was still reported as a failure, not silently counted in.
    assert len(result.skipped_video_failures) == 1


def test_cancel_raises_cancelled_not_failed(deployment_scaffold):
    """
    A user's cancel must not be reported as a crash.

    `load_json_to_database` wraps everything in `except Exception: raise
    RuntimeError(...)`, so without an explicit JobCancelledError guard in
    front of it the worker marks the job failed and the user is told
    something broke when in fact they pressed the button.
    """
    with (
        patch(
            "app.ml.json_pipeline.is_cancel_requested", return_value=True
        ),
        pytest.raises(JobCancelledError),
    ):
        _load(deployment_scaffold, _image_entries(deployment_scaffold))


def test_no_callback_still_loads(deployment_scaffold):
    """The callback is optional; every existing caller omits it."""
    result = _load(deployment_scaffold, _image_entries(deployment_scaffold))
    assert result.total_detections == len(deployment_scaffold["img_paths"])
