"""Tests for the frame-visibility rule.

A video is written to disk as one frame, so only that frame's detections
have a picture. The rule exists in two forms, a SQL predicate and a Python
filter, and the last test here is the one that makes having two safe: it
pins that they select the same detections.
"""

from sqlalchemy import select

from app.ml.detection_visibility import (
    on_visible_frame,
    on_visible_frame_of,
    visible_detections,
)
from app.models import Detection, File
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _video(db, deployment_id, best_frame_number):
    return make_file(
        db,
        deployment_id=deployment_id,
        file_type="video",
        file_format="mp4",
        best_frame_number=best_frame_number,
    )


# ── The Python filter ────────────────────────────────────────────────


def test_every_detection_on_an_image_is_visible(db):
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = make_file(db, deployment_id=dep.id)
    a = make_detection(db, file_id=f.id, confidence=0.9)
    b = make_detection(db, file_id=f.id, confidence=0.4)
    db.commit()

    assert visible_detections(f, [a, b]) == [a, b]


def test_only_the_best_frame_is_visible_on_a_video(db):
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3)
    before = make_detection(db, file_id=f.id, frame_number=1)
    on_best = make_detection(db, file_id=f.id, frame_number=3)
    after = make_detection(db, file_id=f.id, frame_number=7)
    db.commit()

    assert visible_detections(f, [before, on_best, after]) == [on_best]


def test_a_verified_detection_is_visible_on_any_frame(db):
    """A human decision must never end up out of reach. Its thumbnail is
    missing, which is the honest answer, and CropCard degrades."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3)
    off_frame = make_detection(db, file_id=f.id, frame_number=7, verified=True)
    db.commit()

    assert visible_detections(f, [off_frame]) == [off_frame]


def test_a_video_with_no_best_frame_shows_only_verified(db):
    """Frame extraction can fail, and a re-ingested or legacy row may never
    have had one. Such a video has no picture at all, so nothing but a human
    decision survives. The rule must not widen when data is missing."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=None)
    machine = make_detection(db, file_id=f.id, frame_number=2)
    human = make_detection(db, file_id=f.id, frame_number=5, verified=True)
    db.commit()

    assert visible_detections(f, [machine, human]) == [human]


def test_input_order_is_preserved(db):
    """strongest_passing_detection makes stable ordering the caller's
    contract, so the filter must not reorder."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3)
    first = make_detection(db, file_id=f.id, frame_number=3, confidence=0.4)
    second = make_detection(db, file_id=f.id, frame_number=3, confidence=0.9)
    db.commit()

    assert visible_detections(f, [first, second]) == [first, second]
    assert visible_detections(f, [second, first]) == [second, first]


# ── The two lanes agree ──────────────────────────────────────────────


def test_sql_and_python_select_the_same_detections(db):
    """The parity pin. The rule has a SQL form for callers that can filter a
    query and a Python form for callers holding a list. Two implementations
    of one rule can drift; this is what stops it. Covers every branch in one
    fixture set: image, video on and off the best frame, verified off-frame,
    and a video with no best frame at all."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)

    image = make_file(db, deployment_id=dep.id)
    make_detection(db, file_id=image.id, confidence=0.9)
    make_detection(db, file_id=image.id, confidence=0.1)

    video = _video(db, dep.id, best_frame_number=3)
    make_detection(db, file_id=video.id, frame_number=1)
    make_detection(db, file_id=video.id, frame_number=3)
    make_detection(db, file_id=video.id, frame_number=7)
    make_detection(db, file_id=video.id, frame_number=9, verified=True)

    frameless = _video(db, dep.id, best_frame_number=None)
    make_detection(db, file_id=frameless.id, frame_number=2)
    make_detection(db, file_id=frameless.id, frame_number=4, verified=True)
    db.commit()

    for f in (image, video, frameless):
        # Lane 1: the predicate for a query already scoped to this file.
        # It carries only the frame clause on the video branches, so the
        # file_id filter stays.
        scoped = set(
            db.execute(
                select(Detection.id)
                .where(Detection.file_id == f.id)
                .where(on_visible_frame_of(f))
            ).scalars()
        )
        # Lane 2: the Python filter over the same rows.
        in_memory = {
            d.id
            for d in visible_detections(
                f,
                db.execute(
                    select(Detection).where(Detection.file_id == f.id)
                ).scalars().all(),
            )
        }
        assert scoped == in_memory, f.file_type

    # Lane 3: the column form, over every file at once.
    joined = set(
        db.execute(
            select(Detection.id)
            .join(File, File.id == Detection.file_id)
            .where(on_visible_frame())
        ).scalars()
    )
    every_file = set()
    for f in (image, video, frameless):
        every_file |= {
            d.id
            for d in visible_detections(
                f,
                db.execute(
                    select(Detection).where(Detection.file_id == f.id)
                ).scalars().all(),
            )
        }
    assert joined == every_file
