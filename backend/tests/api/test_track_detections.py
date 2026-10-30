"""One animal's frames, for the opened track on the Detections tab.

The grid shows one card per track. Opening that card asks this endpoint
for the boxes behind it, so a person can relabel the part of a track
that followed a different animal without cutting the track in two.

The rule that matters here is which frames come back: those at or above
the classification gate the analysis ran with, or verified. That line is
fixed per analysis, so neither the counting threshold nor the Labels
slider changes what is inside a track. Weaker boxes stay stored, holding
the track together, but there is nothing in them to check.
"""

from app.models import Detection
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
    make_track,
)


def _clip(db, *, counting_threshold=0.2, gate_used=0.1):
    project = make_project(db, counting_threshold=counting_threshold)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id, classification_gate_used=gate_used)
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path="/fake/NIN_T4_drop12.mp4", frame_rate=30.0,
        best_frame_number=60, width_px=1920, height_px=1080,
        duration_seconds=90.0,
    )
    track = make_track(
        db, file_id=video.id, track_key=3, start_frame=30, end_frame=90,
        representative_frame_number=60, frame_count=4,
    )
    return project, video, track


def test_the_frames_come_back_in_frame_order(client, db):
    _, video, track = _clip(db)
    # Deliberately inserted out of order.
    for frame in (90, 30, 60):
        make_detection(db, file_id=video.id, frame_number=frame, track_id=track.id)
    db.commit()

    resp = client.get(f"/api/tracks/{track.id}/detections")

    assert resp.status_code == 200
    body = resp.json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 60, 90]
    assert body["file_id"] == video.id
    assert body["file_name"] == "NIN_T4_drop12.mp4"
    assert body["track"]["track_key"] == 3
    assert body["frame_rate"] == 30.0


def test_only_frames_at_or_above_the_runs_gate_come_back(client, db):
    """A box below the gate is too weak to classify or count; the
    tracker kept it to hold the animal together, not for a person."""
    _, video, track = _clip(db, gate_used=0.1)
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   confidence=0.9)
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                   confidence=0.1)
    make_detection(db, file_id=video.id, frame_number=90, track_id=track.id,
                   confidence=0.05)
    db.commit()

    body = client.get(f"/api/tracks/{track.id}/detections").json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 60]


def test_a_verified_weak_frame_stays_and_a_rejected_one_goes(client, db):
    """The scope rule's verified arm, as everywhere else: a person who
    confirmed a weak box saw an animal there; a weak box rejected by a
    file sign-off stays out at every threshold."""
    _, video, track = _clip(db, gate_used=0.1)
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   confidence=0.9)
    seen = make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                          confidence=0.05, verified=True)
    make_detection(db, file_id=video.id, frame_number=90, track_id=track.id,
                   confidence=0.05, verified=True, label="false detection")
    db.commit()

    body = client.get(f"/api/tracks/{track.id}/detections").json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 60]
    assert body["detections"][1]["id"] == seen.id


def test_the_gate_is_the_runs_not_the_projects(client, db):
    """Editing the project's gate applies to new analyses only, so it must
    not change which frames an analysed track holds."""
    project, video, track = _clip(db, gate_used=0.1)
    project.classification_gate = 0.3
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   confidence=0.9)
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                   confidence=0.2)
    db.commit()

    body = client.get(f"/api/tracks/{track.id}/detections").json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 60]


def test_the_counting_threshold_does_not_move_the_frames(client, db):
    """Refine results moves the counting threshold after the fact; what is
    inside a track must not move with it."""
    _, video, track = _clip(db, counting_threshold=0.8, gate_used=0.1)
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   confidence=0.9)
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                   confidence=0.3)
    db.commit()

    body = client.get(f"/api/tracks/{track.id}/detections").json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 60]


def test_a_tracked_video_without_a_recorded_gate_fails_loudly(client, db):
    """Every analysis since tracks exist records its gate. A missing one is
    a broken record, and guessing a line would hide or show frames
    silently."""
    _, video, track = _clip(db, gate_used=None)
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   confidence=0.9)
    db.commit()

    assert client.get(f"/api/tracks/{track.id}/detections").status_code == 500


def test_each_frame_carries_its_own_crop_box(client, db):
    """The card's overlay. A wide box and a tall one sit differently
    inside their crops, so copying one row's value onto another would
    draw the overlay in the wrong place."""
    _, video, track = _clip(db)
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   bbox_width=0.4, bbox_height=0.1)
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                   bbox_width=0.1, bbox_height=0.4)
    db.commit()

    rows = client.get(f"/api/tracks/{track.id}/detections").json()["detections"]

    assert all(r["crop_bbox"] is not None for r in rows)
    assert rows[0]["crop_bbox"] != rows[1]["crop_bbox"]
    # The wide box fills its crop's width and is short in it.
    assert rows[0]["crop_bbox"]["w"] > rows[0]["crop_bbox"]["h"]


def test_boxes_of_other_tracks_and_other_files_are_not_included(client, db):
    _, video, track = _clip(db)
    mine = make_detection(db, file_id=video.id, frame_number=60, track_id=track.id)
    other = make_track(db, file_id=video.id, track_key=4, start_frame=300,
                       end_frame=360, representative_frame_number=330)
    make_detection(db, file_id=video.id, frame_number=330, track_id=other.id)
    make_detection(db, file_id=video.id, frame_number=44)  # untracked
    db.commit()

    rows = client.get(f"/api/tracks/{track.id}/detections").json()["detections"]

    assert [r["id"] for r in rows] == [mine.id]


def test_an_unknown_track_is_a_404(client, db):
    assert client.get("/api/tracks/no-such-track/detections").status_code == 404


def test_a_file_without_pixel_dimensions_has_no_crop_box(client, db):
    """Honest rather than guessed: without the file's size the crop
    square cannot be worked out, so the card draws no overlay."""
    _, video, track = _clip(db)
    video.width_px = None
    video.height_px = None
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id)
    db.commit()

    rows = client.get(f"/api/tracks/{track.id}/detections").json()["detections"]

    assert len(rows) == 1 and rows[0]["crop_bbox"] is None


def test_a_box_with_no_geometry_is_left_out(client, db):
    """Event-level rows carry no bbox and cannot be a card."""
    _, video, track = _clip(db)
    real = make_detection(db, file_id=video.id, frame_number=30, track_id=track.id)
    boxless = make_detection(db, file_id=video.id, frame_number=60, track_id=track.id)
    db.query(Detection).filter(Detection.id == boxless.id).update(
        {"bbox_x": None, "bbox_y": None, "bbox_width": None, "bbox_height": None}
    )
    db.commit()

    rows = client.get(f"/api/tracks/{track.id}/detections").json()["detections"]

    assert [r["id"] for r in rows] == [real.id]
