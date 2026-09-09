"""A verdict on a track's card is a verdict on the animal.

X, relabel, verify, dismiss and undo on any box of a track reach every
box the tracker followed the animal through, and a file sign-off does
the same for every track in the file. Boxes with no track (images, the
untracked rows of a pre-tracking run) are untouched.
"""

from app.api.crud.detection import expand_to_tracks
from app.models import Detection, File
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
    make_track,
    make_video_box,
)


def _tracked_video(db, *, counting_threshold=0.2):
    """A video with two tracks of three boxes each (representative in the
    middle), plus one untracked legacy box on the cover and one one-frame
    track (what the migration makes of an old hand-drawn box; nothing
    creates these any more, but they sit in databases already out)."""
    project = make_project(db, counting_threshold=counting_threshold)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    video = make_file(
        db, deployment_id=dep.id, file_type="video", file_format="mp4",
        file_path="/fake/clip.mp4", frame_rate=30.0, best_frame_number=60,
    )
    one = make_track(db, file_id=video.id, track_key=1, start_frame=30, end_frame=90,
                     representative_frame_number=60)
    two = make_track(db, file_id=video.id, track_key=2, start_frame=300, end_frame=360,
                     representative_frame_number=330)
    boxes = {
        "one": [make_detection(db, file_id=video.id, frame_number=f, track_id=one.id,
                               category="elasmobranch") for f in (30, 60, 90)],
        "two": [make_detection(db, file_id=video.id, frame_number=f, track_id=two.id,
                               category="elasmobranch") for f in (300, 330, 360)],
        "loose": [make_detection(db, file_id=video.id, frame_number=60,
                                 category="elasmobranch")],
        "drawn": [make_video_box(db, file_id=video.id, frame_number=60, job_id=None,
                                 confidence=1.0, verified=True, category="elasmobranch")],
    }
    db.commit()
    return project, video, one, two, boxes


def _rows(db, ids):
    return {d.id: d for d in db.query(Detection).filter(Detection.id.in_(ids)).all()}


def test_expand_to_tracks_adds_every_sibling_once(db):
    _, _, _, _, boxes = _tracked_video(db)
    card = boxes["one"][1].id
    loose = boxes["loose"][0].id

    expanded = expand_to_tracks(db, [card, loose])

    assert expanded[:2] == [card, loose]
    assert set(expanded) == {card, loose, *(b.id for b in boxes["one"])}
    assert len(expanded) == len(set(expanded))
    assert expand_to_tracks(db, []) == []


def test_relabel_on_the_card_relabels_the_whole_track(client, db):
    _, video, _, _, boxes = _tracked_video(db)
    card = boxes["one"][1]

    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [card.id], "label": "great hammerhead"},
    )
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 3

    db.expire_all()
    for box in boxes["one"]:
        assert (box.label, box.verified, box.classification_method) == (
            "great hammerhead", True, "human"
        )
    # The other track and the loose box are not the same animal.
    assert all(b.label is None for b in boxes["two"] + boxes["loose"])


def test_x_on_the_card_rejects_the_whole_track(client, db):
    _, _, _, _, boxes = _tracked_video(db)
    card = boxes["two"][1]

    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [card.id], "label": "false detection"},
    )
    assert resp.status_code == 200

    db.expire_all()
    assert all(b.label == "false detection" and b.verified for b in boxes["two"])
    assert all(b.label is None and not b.verified for b in boxes["one"])


def test_verify_on_the_card_verifies_the_whole_track(client, db):
    _, _, _, _, boxes = _tracked_video(db)
    card = boxes["one"][1]

    resp = client.patch(f"/api/detections/{card.id}/verify", json={"verified": True})
    assert resp.status_code == 200
    db.expire_all()
    assert all(b.verified for b in boxes["one"])
    assert not any(b.verified for b in boxes["two"])

    resp = client.patch(f"/api/detections/{card.id}/verify", json={"verified": False})
    assert resp.status_code == 200
    db.expire_all()
    assert not any(b.verified for b in boxes["one"])


def test_bulk_verify_dismiss_and_undo_reach_the_track(client, db):
    _, _, _, _, boxes = _tracked_video(db)
    card = boxes["one"][1]

    assert client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [card.id], "verified": True},
    ).json()["updated_count"] == 3
    db.expire_all()
    assert all(b.verified for b in boxes["one"])

    assert client.post(
        "/api/detections/bulk-dismiss",
        json={"detection_ids": [card.id], "dismissed": True},
    ).json()["updated_count"] == 3
    db.expire_all()
    assert all(b.suggestion_dismissed for b in boxes["one"])
    assert not any(b.suggestion_dismissed for b in boxes["two"])

    resp = client.post(
        "/api/detections/bulk-revert-to-original", json={"detection_ids": [card.id]}
    )
    assert resp.status_code == 200
    assert {r["detection_id"] for r in resp.json()["reverted"]} == {b.id for b in boxes["one"]}
    db.expire_all()
    assert not any(b.verified for b in boxes["one"])


def test_a_file_sign_off_reaches_every_track_through_its_card(client, db):
    """Signing the video off is a verdict on every card the Files viewer
    shows as a bar: every track's representative box, wherever in the
    clip it sits. A weak track is rejected, a strong one verified, and
    through the card the verdict reaches every box of the track. A box
    with no card (the untracked legacy row) is left alone."""
    project, video, _, two, boxes = _tracked_video(db, counting_threshold=0.5)
    # Track one's boxes are weak, so the sign-off rejects the whole track.
    for b in boxes["one"]:
        b.confidence = 0.3
    db.commit()

    resp = client.patch(f"/api/files/{video.id}", json={"verified": True})
    assert resp.status_code == 200

    db.expire_all()
    assert all(b.verified and b.label == "false detection" for b in boxes["one"])
    # Track two lives on frames 300 to 360; its card is a bar like any other.
    assert all(b.verified for b in boxes["two"])
    assert all(b.label is None for b in boxes["two"])
    assert not boxes["loose"][0].verified
    assert db.get(File, video.id).verified is True

    # And back: unverify clears every box of the file, tracks included.
    resp = client.patch(f"/api/files/{video.id}", json={"verified": False})
    assert resp.status_code == 200
    db.expire_all()
    assert not any(b.verified for b in boxes["one"] + boxes["two"] + boxes["loose"])


def test_a_verdict_from_any_box_of_a_track_reaches_the_whole_track(client, db):
    """Today's rule, which the module docstring claims and nothing pinned.

    ``expand_to_tracks`` expands whatever id it is handed, not only the
    track's card. The Files viewer relies on this: scrubbing off the cover
    and clicking a box selects that frame's box, and Enter is expected to
    sign off the animal, not one moment of it. Pinned here so the
    ``expand_tracks`` opt-out cannot quietly become the default.
    """
    _, _, _, _, boxes = _tracked_video(db)
    off_card = boxes["one"][0]  # frame 30; the card is frame 60

    resp = client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [off_card.id], "verified": True},
    )
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 3

    db.expire_all()
    assert all(b.verified for b in boxes["one"])
    assert not any(b.verified for b in boxes["two"])


def test_expand_tracks_false_keeps_a_verdict_on_the_boxes_named(client, db):
    """The opt-out the opened track uses.

    Inside an opened track a person judges frames, not the animal, so a
    verdict must stay on what they picked. That includes the track's own
    representative box, which is one card among its neighbours there and
    must not behave differently from them.
    """
    _, _, _, _, boxes = _tracked_video(db)
    off_card, card = boxes["one"][0], boxes["one"][1]

    resp = client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [off_card.id], "verified": True, "expand_tracks": False},
    )
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 1
    db.expire_all()
    assert off_card.verified and not card.verified and not boxes["one"][2].verified

    # The card itself is no exception inside an opened track.
    resp = client.post(
        "/api/detections/bulk-relabel",
        json={
            "detection_ids": [card.id],
            "label": "great hammerhead",
            "expand_tracks": False,
        },
    )
    assert resp.json()["updated_count"] == 1
    db.expire_all()
    assert card.label == "great hammerhead"
    assert boxes["one"][0].label is None and boxes["one"][2].label is None


def test_boxes_without_a_track_are_left_alone(client, db):
    _, _, _, _, boxes = _tracked_video(db)
    loose = boxes["loose"][0]

    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [loose.id], "label": "nurse shark"},
    )
    assert resp.json()["updated_count"] == 1
    db.expire_all()
    assert loose.label == "nurse shark"
    assert all(b.label is None for b in boxes["one"] + boxes["two"])


def test_a_species_label_keeps_the_detectors_category(client, db):
    """The picker sends "animal" beside every species label. That must
    not rename an elasmobranch box: the category is the detector's. A
    person box does move to the wildlife category with the label, and a
    category-only relabel still applies."""
    _, video, _, _, boxes = _tracked_video(db)
    person = make_detection(db, file_id=video.id, frame_number=60, category="person")
    db.commit()

    client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [boxes["one"][1].id], "label": "nurse shark",
              "category": "animal"},
    )
    client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [person.id], "label": "nurse shark", "category": "animal"},
    )
    client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [boxes["loose"][0].id], "category": "vehicle"},
    )
    db.expire_all()
    assert all(b.category == "elasmobranch" and b.label == "nurse shark" for b in boxes["one"])
    assert (person.category, person.label) == ("animal", "nurse shark")
    assert boxes["loose"][0].category == "vehicle"


def test_a_box_cannot_be_drawn_on_a_clip(client, db):
    """Photos only. A hand-drawn box lands on the one frame on screen,
    which on an hour of footage is one frame in ten thousand, so it could
    never be how a missed animal is recorded; the Counts page is. The
    refusal is a 400 that says where to go instead, and it leaves the
    clip's own boxes alone."""
    _, video, _, _, _ = _tracked_video(db)
    before = db.query(Detection).filter(Detection.file_id == video.id).count()

    resp = client.post(
        "/api/detections",
        json={
            "file_id": video.id, "category": "animal", "frame_number": 60,
            "bbox_x": 0.5, "bbox_y": 0.5, "bbox_width": 0.1, "bbox_height": 0.1,
        },
    )

    assert resp.status_code == 400, resp.text
    assert "Counts page" in resp.json()["detail"]
    assert db.query(Detection).filter(Detection.file_id == video.id).count() == before


def test_a_box_can_still_be_drawn_on_a_photo(client, db):
    """The refusal is about clips, not about drawing."""
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    photo = make_file(db, deployment_id=dep.id)
    db.commit()

    resp = client.post(
        "/api/detections",
        json={
            "file_id": photo.id, "category": "animal",
            "bbox_x": 0.5, "bbox_y": 0.5, "bbox_width": 0.1, "bbox_height": 0.1,
        },
    )

    assert resp.status_code == 201, resp.text
    drawn = db.get(Detection, resp.json()["id"])
    assert drawn.verified is True and drawn.track_id is None
