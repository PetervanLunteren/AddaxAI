"""One animal's frames, for the opened track on the Detections tab.

The grid shows one card per track. Opening that card asks this endpoint
for the boxes behind it, so a person can relabel the part of a track
that followed a different animal without cutting the track in two.

The rule that matters here is the confidence scope. The Labels slider
digs *down* below the project's counting threshold, so a card visible
at a lowered slider has to open to the frames that were visible with
it. Gating at the project threshold alone opens an empty track under a
card the person can plainly see.
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


def _clip(db, *, counting_threshold=0.2):
    project = make_project(db, counting_threshold=counting_threshold)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
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


def test_weak_boxes_are_hidden_until_the_slider_reaches_them(client, db):
    """The scope is the grid's, and the slider digs down.

    The tracker keeps boxes far below the counting threshold, so a track
    routinely holds more frames than the grid shows. Which of them
    appear must follow the same slider the grid was sorted at.
    """
    _, video, track = _clip(db, counting_threshold=0.2)
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   confidence=0.9)
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                   confidence=0.5)
    weak = make_detection(db, file_id=video.id, frame_number=90, track_id=track.id,
                          confidence=0.05)
    db.commit()

    # At the project threshold the weak box is out, as everywhere else.
    body = client.get(f"/api/tracks/{track.id}/detections").json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 60]

    # Slider dragged down below it: the card was visible there, so its
    # frames must be too.
    body = client.get(
        f"/api/tracks/{track.id}/detections", params={"min_confidence": 0.01}
    ).json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 60, 90]
    assert weak.id in {d["id"] for d in body["detections"]}

    # A slider above the threshold filters literally, as the grid does.
    body = client.get(
        f"/api/tracks/{track.id}/detections", params={"min_confidence": 0.7}
    ).json()
    assert [d["frame_number"] for d in body["detections"]] == [30]


def test_a_rejected_weak_box_is_out_at_the_threshold(client, db):
    """Verifying a file marks its invisible weak boxes "false detection".

    Those rows are excluded by the verified arm of the scope rule, not by
    a rule of their own: a rejected box below the floor fails the
    confidence arm and is disqualified from the verified arm by its
    label. Drop the slider under its confidence and it reappears, which
    is what the grid does too, because the person is then explicitly
    asking to see the low-confidence tail.
    """
    _, video, track = _clip(db, counting_threshold=0.2)
    make_detection(db, file_id=video.id, frame_number=30, track_id=track.id,
                   confidence=0.9)
    make_detection(db, file_id=video.id, frame_number=60, track_id=track.id,
                   confidence=0.03, verified=True, label="false detection")
    # A verified box above the floor keeps its verdict and stays visible.
    make_detection(db, file_id=video.id, frame_number=90, track_id=track.id,
                   confidence=0.4, verified=True, label="false detection")
    db.commit()

    body = client.get(f"/api/tracks/{track.id}/detections").json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 90]

    body = client.get(
        f"/api/tracks/{track.id}/detections", params={"min_confidence": 0.01}
    ).json()
    assert [d["frame_number"] for d in body["detections"]] == [30, 60, 90]


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
