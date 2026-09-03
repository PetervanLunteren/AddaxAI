"""Verifying a file rejects its invisible boxes.

Signing a file off says "the boxes you can see are all there is". Every
box below the threshold on the frame the person saw is therefore wrong,
so it is rejected the way the X key rejects a box: marked "false
detection" and verified. Every box they could see is verified as it
stands. One rule for empty and non-empty files alike.

Leaving the weak boxes untouched made "verified" true only at the
threshold it was checked at: drop the confidence slider and the file
came back carrying a 3% smudge while still flagged verified. Rejected
rows cannot come back that way, because `threshold_or_verified`
(`ml/label_exclusion.py`) keeps a below-threshold non-label box out of
every user-facing scope at any threshold.

They used to be DELETED instead of rejected. That reported them as
missing on every reprocess of an unticked file, and nothing could ever
bring them back; kept as rejected rows they still match their
``results.json`` boxes, and an unverify hands them to the machine to
restore on the next reprocess.
"""

from datetime import datetime

from sqlalchemy import select

from app.models import Detection, Event, EventObservation, File
from app.models.event import event_files
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def _file_in_project(db, threshold=0.2, **file_kw):
    p = make_project(db, counting_threshold=threshold)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    return make_file(db, deployment_id=d.id, **file_kw)


def _verify(client, file_id, verified=True):
    resp = client.patch(f"/api/files/{file_id}", json={"verified": verified})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _boxes(db, file_id):
    return db.query(Detection).filter(Detection.file_id == file_id).all()


def _assert_rejected(det):
    assert det.label == "false detection"
    assert det.verified is True
    assert det.classification_method == "human"


def test_verifying_an_empty_file_rejects_the_detector_boxes(client, db):
    f = _file_in_project(db)
    for conf in (0.01, 0.05, 0.19):
        make_detection(db, file_id=f.id, confidence=conf)
    db.commit()

    _verify(client, f.id)

    boxes = _boxes(db, f.id)
    assert len(boxes) == 3, "rejected, not deleted"
    for det in boxes:
        _assert_rejected(det)
    db.refresh(f)
    assert f.verified is True


def test_unverifying_leaves_the_rejects_for_the_machine(client, db):
    """Taking the sign-off back unverifies every box, the rejected ones
    included. Their "false detection" label stays for now, but unverified
    it belongs to the machine again: the next reprocess overwrites it
    with the AI's own call, which is the whole point of keeping the row."""
    f = _file_in_project(db)
    make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    _verify(client, f.id)
    _verify(client, f.id, verified=False)

    (det,) = _boxes(db, f.id)
    assert det.verified is False
    assert det.label == "false detection"
    db.refresh(f)
    assert f.verified is False


def test_verifying_a_file_with_a_passing_box_rejects_the_weak_ones(client, db):
    """Same rule on a file that is not empty: the box the person saw is
    signed off as it stands, the one they could not see is rejected."""
    f = _file_in_project(db)
    strong = make_detection(db, file_id=f.id, confidence=0.9, label="deer")
    weak = make_detection(db, file_id=f.id, confidence=0.05, label="deer")
    db.commit()

    _verify(client, f.id)

    db.refresh(strong)
    assert strong.verified is True
    assert strong.label == "deer"
    db.refresh(weak)
    _assert_rejected(weak)
    db.refresh(f)
    assert f.verified is True


def test_the_verify_write_is_frame_gated(client, db):
    """A video is judged on its best frame. Boxes on the frames nobody saw
    are neither rejected nor signed off, strong or weak. Before this the
    verify write had no frame clause and signed off boxes on frames the
    person never opened."""
    f = _file_in_project(db, file_type="video", best_frame_number=0)
    on_weak = make_detection(db, file_id=f.id, confidence=0.05, frame_number=0).id
    on_strong = make_detection(db, file_id=f.id, confidence=0.9, frame_number=0).id
    off_weak = make_detection(db, file_id=f.id, confidence=0.05, frame_number=7).id
    off_strong = make_detection(db, file_id=f.id, confidence=0.9, frame_number=7).id
    db.commit()

    _verify(client, f.id)

    by_id = {d.id: d for d in _boxes(db, f.id)}
    _assert_rejected(by_id[on_weak])
    assert by_id[on_strong].verified is True
    assert by_id[on_strong].label is None
    assert by_id[off_weak].verified is False
    assert by_id[off_weak].label is None
    assert by_id[off_strong].verified is False


def test_verifying_a_file_recomputes_what_it_is_about(client, db):
    """Rejecting boxes and verifying boxes both move the strongest passing
    detection, so `observation_type` is re-derived on the spot, as the
    detection endpoints do. It used to stay stale until the next reprocess."""
    f = _file_in_project(db, observation_type="animal")
    make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    _verify(client, f.id)
    db.refresh(f)
    assert f.observation_type == "blank"

    g = _file_in_project(db, observation_type="unclassified")
    make_detection(db, file_id=g.id, confidence=0.9, category="person")
    db.commit()

    _verify(client, g.id)
    db.refresh(g)
    assert g.observation_type == "person"


def test_unverifying_a_file_recomputes_the_counts(client, db):
    """A verified weak box counts; taking the file's sign-off back clears
    the box's verified flag, so the observation it produced goes and a
    confirmed event is unconfirmed. Neither happened before: the file
    router recomputed nothing."""
    p = make_project(db, counting_threshold=0.2)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = make_event_with_files(
        db, deployment_id=d.id, event_start_local=datetime(2024, 1, 1, 12, 0)
    )
    file_id = db.execute(
        select(event_files.c.file_id).where(event_files.c.event_id == ev.id)
    ).scalar_one()
    weak = make_detection(db, file_id=file_id, confidence=0.05, label="deer")
    db.commit()

    resp = client.patch(f"/api/detections/{weak.id}/verify", json={"verified": True})
    assert resp.status_code == 200, resp.text
    assert db.query(EventObservation).filter_by(event_id=ev.id).count() == 1
    db.get(Event, ev.id).confirmed = True
    db.commit()

    _verify(client, file_id, verified=False)

    db.expire_all()
    assert db.get(File, file_id).verified is False
    assert db.get(Detection, weak.id).verified is False
    assert db.query(EventObservation).filter_by(event_id=ev.id).count() == 0
    assert db.get(Event, ev.id).confirmed is False


def test_reverifying_picks_up_boxes_added_since(client, db):
    """Verify is idempotent, not a one-shot toggle. A box that arrived
    after the first sign-off (a reprocess, a lowered threshold) is signed
    off by the next Enter, where before that Enter was a no-op."""
    f = _file_in_project(db)
    make_detection(db, file_id=f.id, confidence=0.9)
    db.commit()
    _verify(client, f.id)

    later = make_detection(db, file_id=f.id, confidence=0.9)
    db.commit()
    db.refresh(later)
    assert later.verified is False

    _verify(client, f.id)
    db.refresh(later)
    assert later.verified is True


def test_only_boxes_on_the_frame_the_person_saw_are_rejected(client, db):
    """A clip reads empty on its best frame alone, and that frame is the
    only one the app shows. Boxes on the other sampled frames were never
    on screen, so a verdict about the visible frame is not a verdict
    about them."""
    f = _file_in_project(db, file_type="video", best_frame_number=0)
    on_screen = make_detection(
        db, file_id=f.id, confidence=0.05, frame_number=0
    ).id
    off_screen = make_detection(
        db, file_id=f.id, confidence=0.95, frame_number=7
    ).id
    db.commit()

    _verify(client, f.id)

    by_id = {d.id: d for d in _boxes(db, f.id)}
    _assert_rejected(by_id[on_screen])
    assert by_id[off_screen].label is None
    assert by_id[off_screen].verified is False


def test_a_drawn_box_survives_a_file_verify(client, db):
    """The rule never reads who drew a box, and it does not have to: a
    drawn box is verified at confidence 1.0, so it is always visible and
    never below the threshold. Here it sits off the best frame, where
    nothing is touched at all, so the noise beside it is untouched too.
    """
    f = _file_in_project(db, file_type="video", best_frame_number=0)
    drawn = make_detection(
        db,
        file_id=f.id,
        confidence=1.0,
        frame_number=5,
        verified=True,
        classification_method="human",
        label="deer",
    )
    noise = make_detection(db, file_id=f.id, confidence=0.05, frame_number=5)
    db.commit()

    _verify(client, f.id)

    db.refresh(drawn)
    assert drawn.label == "deer"
    assert drawn.verified is True
    db.refresh(noise)
    assert noise.label is None
    assert noise.verified is False


def test_a_rejected_box_does_not_make_a_file_look_occupied(client, db):
    """A file whose only real box was marked false reads as empty in the
    Files tab (`is_a_real_detection()`), and verifying it rejects the weak
    boxes beside the rejected one. The box the person X'd is verified
    already, so the rule never touches it.

    The untick in the middle mirrors the user's route: marking a box
    false verifies it, which rolls up and leaves the file signed off, so
    a person unticks it to look again and then signs it off as empty.
    """
    f = _file_in_project(db)
    rejected = make_detection(db, file_id=f.id, confidence=0.85)
    noise = make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [rejected.id], "label": "false detection"},
    )
    assert resp.status_code == 200, resp.text

    project_id = f.deployment.project_id
    empties = client.get(
        f"/api/projects/{project_id}/labels/files", params={"empty": "show_only"}
    ).json()
    assert empties["total"] == 1, "the file reads as empty"

    _verify(client, f.id, verified=False)
    _verify(client, f.id)

    db.refresh(noise)
    _assert_rejected(noise)


def test_an_auto_rejected_box_matches_a_pressed_x(client, db):
    """The parity pin: a weak box rejected by a file sign-off must be
    indistinguishable from a box the person pressed X on, field for
    field. Two code paths write the verdict (`mark_detections_false` and
    the bulk-relabel endpoint); this is what keeps them from drifting."""
    p = make_project(db, counting_threshold=0.2)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id)
    auto = make_detection(db, file_id=f.id, confidence=0.05)
    pressed = make_detection(db, file_id=f.id, confidence=0.85)
    db.commit()

    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [pressed.id], "label": "false detection"},
    )
    assert resp.status_code == 200, resp.text
    _verify(client, f.id, verified=False)
    _verify(client, f.id)

    db.refresh(auto)
    db.refresh(pressed)
    for field in (
        "label",
        "label_confidence",
        "label_taxonomy_id",
        "common_name",
        "scientific_name",
        "classification_method",
        "verified",
    ):
        assert getattr(auto, field) == getattr(pressed, field), field
    assert auto.label_taxonomy_id is not None, (
        "both share the project's custom 'false detection' taxonomy row"
    )


def test_a_rejected_weak_box_stays_out_of_every_scope(client, db):
    """The other half of the design: rejecting instead of deleting only
    works because `threshold_or_verified` keeps the rejected row buried.
    Verified rows used to pass every user-facing scope unconditionally,
    so without the refinement a signed-off project surfaced thousands of
    them (4,050 weak boxes against 7,526 passing ones on a real one)."""
    f = _file_in_project(db)
    make_detection(db, file_id=f.id, confidence=0.9, label="deer")
    make_detection(db, file_id=f.id, confidence=0.05, label="deer")
    db.commit()
    project_id = f.deployment.project_id

    _verify(client, f.id)

    count = client.get(
        f"/api/projects/{project_id}/detection-count", params={"threshold": 0.2}
    ).json()["count"]
    assert count == 1, "the rejected weak box is not a countable detection"

    stats = client.get(
        f"/api/projects/{project_id}/label-stats", params={"threshold": 0.2}
    ).json()
    assert {row["label"] for row in stats} == {"deer"}, (
        "'false detection' is not offered as a label"
    )
