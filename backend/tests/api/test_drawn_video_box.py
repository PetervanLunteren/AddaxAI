"""A box drawn on a clip's frame is a one-frame track of its own.

Image and video verification feel the same: a clip's still is a canvas
like a photo (2026-11-03). What the backend adds is the track, so every
surface that reads tracks and cards (MaxN per file and per event, the
tile chips, the timeline, the sign-off, the exports) sees the drawn box
without a rule of its own. Its key counts down from -1, out of the
tracker's reach on a re-ingest.
"""

from datetime import datetime

from app.api.crud.event_observation import (
    calculate_max_n_for_event,
    list_event_observations,
)
from app.models import Detection, Track
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
    make_track,
)

BOX = {
    "category": "animal",
    "bbox_x": 0.5, "bbox_y": 0.5, "bbox_width": 0.1, "bbox_height": 0.1,
}


def _clip(db, *, tracks=(1, 2)):
    """A clip with the tracker's tracks 1 and 2 (three boxes each, the
    card in the middle, a crop on disk) and its cover on frame 60."""
    project = make_project(db, counting_threshold=0.2)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path="/fake/clip.mp4", frame_rate=30.0, best_frame_number=60,
    )
    for key, frames in zip(tracks, ((30, 60, 90), (300, 330, 360)), strict=False):
        track = make_track(
            db, file_id=video.id, track_key=key, start_frame=frames[0],
            end_frame=frames[-1], representative_frame_number=frames[1],
            crop_path=f"/fake/track{key:06d}.jpg",
        )
        for frame in frames:
            make_detection(db, file_id=video.id, frame_number=frame, track_id=track.id,
                           category="elasmobranch", confidence=0.9)
    db.commit()
    return project, video


def _draw(client, video, frame=60, **extra):
    resp = client.post(
        "/api/detections", json={"file_id": video.id, "frame_number": frame, **BOX, **extra}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_a_box_drawn_on_a_clip_is_a_one_frame_track_with_a_negative_key(client, db):
    """The track's one frame is the one the box was drawn on, so the box
    is the track's card; the box is the person's, verified, at 1.0."""
    _, video = _clip(db)

    created = _draw(client, video)

    box = db.get(Detection, created["id"])
    track = box.track
    assert track is not None and created["track_id"] == track.id
    assert track.track_key == -1
    assert (track.start_frame, track.end_frame, track.representative_frame_number) == (60, 60, 60)
    assert track.frame_count == 1
    assert track.max_confidence == 1.0
    assert track.crop_path is None
    assert box.job_id is None and box.verified and box.classification_method == "human"


def test_each_drawn_box_takes_the_next_key_down(client, db):
    """-1, -2, ... per clip, whatever the tracker's keys are: a clip with
    tracks 1 and 2 and a clip with none both start at -1."""
    _, video = _clip(db)
    keys = [
        db.get(Detection, _draw(client, video, frame=f)["id"]).track.track_key
        for f in (60, 90)
    ]
    assert keys == [-1, -2]

    _, bare = _clip(db, tracks=())
    assert db.get(Detection, _draw(client, bare)["id"]).track.track_key == -1


def test_a_box_on_a_clip_without_a_frame_is_refused(client, db):
    """A one-frame track needs its frame. The viewer always sends the
    frame on screen; a request without one is a 400 that creates nothing."""
    _, video = _clip(db)
    before = db.query(Detection).filter(Detection.file_id == video.id).count()

    resp = client.post("/api/detections", json={"file_id": video.id, **BOX})

    assert resp.status_code == 400, resp.text
    assert "frame" in resp.json()["detail"]
    db.expire_all()
    assert db.query(Detection).filter(Detection.file_id == video.id).count() == before
    assert db.query(Track).filter(Track.file_id == video.id).count() == 2


def test_a_drawn_box_is_a_card_of_the_clip_and_raises_max_at_once_on_its_frame(client, db):
    """The cards scope the tiles and the viewer read carries the box and
    its track, and the clip's MaxN for that species on that frame is one
    more: a second elasmobranch beside the tracker's on the cover."""
    _, video = _clip(db)
    before = client.get(f"/api/files/{video.id}", params={"scope": "cards"}).json()

    created = _draw(client, video, category="elasmobranch")

    after = client.get(f"/api/files/{video.id}", params={"scope": "cards"}).json()
    assert created["id"] in {d["id"] for d in after["detections"]}
    assert created["track_id"] in {t["id"] for t in after["tracks"]}

    def elasmobranchs(payload):
        return [(m["count"], m["frame_number"]) for m in payload["max_at_once"]
                if m["category"] == "elasmobranch"]

    # One per frame before, first reached on frame 30; two on the cover
    # after, which is now the frame that reached the most.
    assert elasmobranchs(before) == [(1, 30)]
    assert elasmobranchs(after) == [(2, 60)]


def test_a_labelled_drawn_box_puts_its_species_in_the_event_count(client, db):
    """The event's MaxN admits a video species through a card. A drawn box
    is its track's card and the person's own label, so a species the AI
    never saw gets its row with a count of one."""
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    ev = make_event_with_files(db, deployment_id=dep.id,
                               event_start_local=datetime(2024, 1, 1, 12))
    video = ev.files[0]
    video.file_type, video.file_format, video.frame_rate, video.best_frame_number = (
        "video", "mp4", 30.0, 60,
    )
    db.commit()
    calculate_max_n_for_event(db, ev.id, project.counting_threshold)
    db.commit()
    assert list_event_observations(db, ev.id) == []

    _draw(client, video, label="turtle")

    db.expire_all()
    assert [(o.label, o.max_n) for o in list_event_observations(db, ev.id)] == [("turtle", 1)]


def test_deleting_a_drawn_box_drops_its_empty_track(client, db):
    """Nothing in the app deletes a box, but a direct API delete must not
    leave a track with no frames behind."""
    _, video = _clip(db)
    created = _draw(client, video)

    assert client.delete(f"/api/detections/{created['id']}").status_code == 204

    db.expire_all()
    assert db.get(Detection, created["id"]) is None
    assert db.get(Track, created["track_id"]) is None


def test_a_sign_off_keeps_a_drawn_box_verified_and_unverify_clears_it(client, db):
    """A drawn box is a card, so the clip's sign-off and its undo reach
    it like any tracked animal's card."""
    _, video = _clip(db)
    created = _draw(client, video)

    assert client.patch(f"/api/files/{video.id}", json={"verified": True}).status_code == 200
    db.expire_all()
    assert db.get(Detection, created["id"]).verified

    assert client.patch(f"/api/files/{video.id}", json={"verified": False}).status_code == 200
    db.expire_all()
    assert not db.get(Detection, created["id"]).verified


def test_moving_a_cards_box_drops_the_stored_crop_and_a_siblings_move_does_not(client, db):
    """The crop the tracking script wrote is the old box's pixels. Once
    the card's box moves, the card is cut from the frame at the new box
    on request; a box on another frame of the track leaves the card's
    crop alone, and so does a label change on the card."""
    _, video = _clip(db)
    track = db.query(Track).filter_by(file_id=video.id, track_key=1).one()
    card = next(b for b in track.detections if b.frame_number == 60)
    sibling = next(b for b in track.detections if b.frame_number == 30)

    assert client.patch(f"/api/detections/{sibling.id}", json={"bbox_x": 0.3}).status_code == 200
    assert client.patch(f"/api/detections/{card.id}", json={"label": "ray"}).status_code == 200
    db.expire_all()
    assert track.crop_path == "/fake/track000001.jpg"

    assert client.patch(f"/api/detections/{card.id}", json={"bbox_x": 0.3}).status_code == 200
    db.expire_all()
    assert track.crop_path is None
