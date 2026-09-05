"""The tracks table and its two foreign keys.

A track goes with its video (CASCADE from files); a box outlives its
track row (SET NULL from tracks), so nothing a person judged can vanish
because a track row did. Both are database rules, so the tests query the
database after each delete rather than reading the session's cache (see
"Deleting analysis data" in DEVELOPERS.md).
"""

from sqlalchemy import select, text

from app.models import Detection, Track
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_track,
)


def _video(db):
    project = make_project(db)
    deployment = make_deployment(db, project_id=project.id)
    return make_file(
        db,
        deployment_id=deployment.id,
        file_type="video",
        file_format="mp4",
        file_path="/fake/clip.mp4",
        frame_rate=30.0,
        best_frame_number=90,
    )


def test_a_track_links_its_boxes_and_names_its_representative(db):
    video = _video(db)
    track = make_track(db, file_id=video.id, track_key=1, representative_frame_number=60)
    boxes = [
        make_detection(db, file_id=video.id, frame_number=f, track_id=track.id)
        for f in (30, 60, 90)
    ]
    db.flush()
    db.expire_all()

    assert sorted(d.frame_number for d in track.detections) == [30, 60, 90]
    representative = [d for d in boxes if d.frame_number == track.representative_frame_number]
    assert len(representative) == 1
    assert representative[0].track is track


def test_two_tracks_in_one_video_cannot_share_a_key(db):
    import pytest
    from sqlalchemy.exc import IntegrityError

    video = _video(db)
    make_track(db, file_id=video.id, track_key=7)
    with pytest.raises(IntegrityError):
        make_track(db, file_id=video.id, track_key=7)
    db.rollback()


def test_deleting_the_video_deletes_its_tracks(db):
    video = _video(db)
    track = make_track(db, file_id=video.id, track_key=1)
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id)
    video_id, track_id = video.id, track.id
    db.commit()

    db.execute(text("DELETE FROM files WHERE id = :id"), {"id": video_id})
    db.commit()

    # Query columns by captured ids: the session still holds the deleted
    # objects and touching one raises rather than returning nothing.
    assert db.execute(select(Track.id).where(Track.id == track_id)).first() is None
    assert (
        db.execute(select(Detection.id).where(Detection.file_id == video_id)).first()
        is None
    )


def test_deleting_a_track_keeps_its_boxes(db):
    video = _video(db)
    track = make_track(db, file_id=video.id, track_key=1)
    box = make_detection(db, file_id=video.id, frame_number=60, track_id=track.id)
    db.commit()

    db.execute(text("DELETE FROM tracks WHERE id = :id"), {"id": track.id})
    db.commit()

    row = db.execute(
        select(Detection.track_id).where(Detection.id == box.id)
    ).one()
    assert row.track_id is None
