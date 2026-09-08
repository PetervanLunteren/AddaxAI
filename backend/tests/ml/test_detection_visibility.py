"""Tests for the visibility rule.

Every image box is visible. A video box is visible on its track's
representative frame and nowhere else. The rule exists in two forms, a
SQL predicate and a Python filter, and the last test here is the one
that makes having two safe: it pins that they select the same
detections.
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
    make_track,
    make_video_box,
)


def _video(db, deployment_id, best_frame_number=3):
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


def test_a_tracks_representative_box_is_visible_and_its_siblings_are_not(db):
    """One card per track, on the frame of its best box. The sibling on
    the cover frame is the same animal, not a second card."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=30)
    track = make_track(db, file_id=f.id, start_frame=30, end_frame=90,
                       representative_frame_number=60)
    on_cover = make_detection(db, file_id=f.id, frame_number=30, track_id=track.id)
    card = make_detection(db, file_id=f.id, frame_number=60, track_id=track.id)
    after = make_detection(db, file_id=f.id, frame_number=90, track_id=track.id)
    db.commit()

    assert visible_detections(f, [on_cover, card, after]) == [card]


def test_an_untracked_video_box_is_visible_nowhere(db):
    """The unverified off-cover boxes of a run analysed before tracking
    became the standard: rows nothing shows, kept for the reprocess
    matcher. The cover and a verdict change nothing; a card needs a
    track."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3)
    on_cover = make_detection(db, file_id=f.id, frame_number=3)
    verified = make_detection(db, file_id=f.id, frame_number=7, verified=True)
    db.commit()

    assert visible_detections(f, [on_cover, verified]) == []


def test_a_one_frame_track_is_a_card_on_its_frame(db):
    """A drawn box and a migrated legacy box: one box, one track, one
    card, on whatever frame it sits."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3)
    drawn = make_video_box(db, file_id=f.id, frame_number=7, verified=True)
    db.commit()

    assert visible_detections(f, [drawn]) == [drawn]


def test_input_order_is_preserved(db):
    """strongest_passing_detection makes stable ordering the caller's
    contract, so the filter must not reorder."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)
    f = _video(db, dep.id, best_frame_number=3)
    first = make_video_box(db, file_id=f.id, frame_number=3, confidence=0.4)
    second = make_video_box(db, file_id=f.id, frame_number=3, confidence=0.9)
    db.commit()

    assert visible_detections(f, [first, second]) == [first, second]
    assert visible_detections(f, [second, first]) == [second, first]


# ── The two lanes agree ──────────────────────────────────────────────


def test_sql_and_python_select_the_same_detections(db):
    """The parity pin. The rule has a SQL form for callers that can filter a
    query and a Python form for callers holding a list. Two implementations
    of one rule can drift; this is what stops it. Covers every branch in one
    fixture set: image, a multi-frame track with a sibling on the cover, a
    one-frame track, untracked boxes on and off the cover, a video with no
    cover at all."""
    project = make_project(db)
    dep = make_deployment(db, project_id=project.id)

    image = make_file(db, deployment_id=dep.id)
    make_detection(db, file_id=image.id, confidence=0.9)
    make_detection(db, file_id=image.id, confidence=0.1)

    video = _video(db, dep.id, best_frame_number=30)
    track = make_track(db, file_id=video.id, start_frame=30, end_frame=90,
                       representative_frame_number=60)
    for frame in (30, 60, 90):
        make_detection(db, file_id=video.id, frame_number=frame, track_id=track.id)
    make_video_box(db, file_id=video.id, frame_number=200, verified=True)
    make_detection(db, file_id=video.id, frame_number=30)  # untracked, on the cover
    make_detection(db, file_id=video.id, frame_number=7, verified=True)  # untracked

    no_cover = _video(db, dep.id, best_frame_number=None)
    make_detection(db, file_id=no_cover.id, frame_number=2)
    make_video_box(db, file_id=no_cover.id, frame_number=5)
    db.commit()

    for f in (image, video, no_cover):
        dets = db.query(Detection).filter(Detection.file_id == f.id).all()
        python_ids = {d.id for d in visible_detections(f, dets)}
        for clause in (on_visible_frame_of(f), on_visible_frame()):
            stmt = (
                select(Detection.id)
                .join(File, File.id == Detection.file_id)
                .where(Detection.file_id == f.id, clause)
            )
            sql_ids = set(db.execute(stmt).scalars().all())
            assert sql_ids == python_ids, f

    # And the numbers themselves, so the fixture cannot silently shrink.
    video_dets = db.query(Detection).filter(Detection.file_id == video.id).all()
    assert len(visible_detections(video, video_dets)) == 2
    no_cover_dets = db.query(Detection).filter(Detection.file_id == no_cover.id).all()
    assert len(visible_detections(no_cover, no_cover_dets)) == 1
