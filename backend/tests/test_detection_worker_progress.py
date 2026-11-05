"""What a deployment analysis says about itself while it runs.

The progress dialog builds its phase rows from the deployment context on
each message, and the job row carries an overall percent for readers
without the socket. Both were wrong on 2026-11-04: a reload dropped the
Embedding row, and the job row read 0 for three and a half hours.
"""

import pytest

from app.workers.detection_worker import (
    _deployment_context,
    _deployment_fraction,
    _run_phases,
)


def test_every_progress_message_carries_the_whole_context():
    """A reconnecting dialog rebuilds its rows from whichever message the
    backend replays, so none may leave out a flag."""
    ctx = _deployment_context(1, 2, 7, 0, has_classifier=False, has_embedding=True)
    assert ctx == {
        "deployment_index": 1,
        "total_deployments": 2,
        "video_count": 7,
        "image_count": 0,
        "has_classifier": False,
        "has_embedding": True,
    }


def test_the_phases_follow_what_the_deployment_runs():
    assert _run_phases(
        has_videos=True, has_images=False, has_classifier=False, has_embedding=True
    ) == ["video_detection", "video_frame_selection", "saving", "postprocessing", "embedding"]
    assert _run_phases(
        has_videos=True, has_images=True, has_classifier=True, has_embedding=False
    ) == [
        "video_detection", "video_frame_selection", "video_classification",
        "image_detection", "image_classification", "saving", "postprocessing",
    ]


def test_detection_is_most_of_a_video_run_and_the_percent_only_grows():
    """The CFD drop: detection was 3 h 25 min of 3 h 28 min, so halfway
    through detection the run is close to halfway, and every later phase
    starts where the one before ended."""
    phases = _run_phases(
        has_videos=True, has_images=False, has_classifier=False, has_embedding=True
    )
    assert _deployment_fraction("video_detection", 0.5, phases) == pytest.approx(0.45)

    steps = [("init", 0.0)] + [(p, f) for p in phases for f in (0.0, 0.5, 1.0)]
    values = [_deployment_fraction(p, f, phases) for p, f in steps]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] == pytest.approx(1.0)
    assert _deployment_fraction("finalize", 0.0, phases) == 1.0


def test_a_phase_progress_outside_zero_to_one_is_clamped():
    phases = ["video_detection", "saving"]
    assert _deployment_fraction("video_detection", 1.7, phases) == pytest.approx(90 / 93)
    assert _deployment_fraction("video_detection", -1.0, phases) == 0.0
