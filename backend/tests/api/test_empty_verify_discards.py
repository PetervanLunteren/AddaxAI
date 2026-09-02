"""Verifying a file throws its invisible boxes away.

Signing a file off says "the boxes you can see are all there is". Every
box below the threshold on the frame the person saw is therefore wrong,
so it is deleted rather than kept, and every box they could see is
verified. One rule for empty and non-empty files alike.

Keeping the weak boxes made "verified" true only at the threshold it was
checked at. Drop the confidence slider and the file came back carrying a
3% smudge while still flagged verified; raise the project threshold
afterwards and it exported ``is_verified = TRUE`` beside a species nobody
had confirmed.

This is only defensible because the Files viewer draws no sub-threshold
boxes, so the person is judging the picture and not a threshold. If weak
boxes are ever drawn there, these tests should be revisited along with
the rule (``crud.file.set_file_verified``).
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


def test_verifying_an_empty_file_discards_the_detector_boxes(client, db):
    f = _file_in_project(db)
    for conf in (0.01, 0.05, 0.19):
        make_detection(db, file_id=f.id, confidence=conf)
    db.commit()
    assert len(_boxes(db, f.id)) == 3

    _verify(client, f.id)

    assert _boxes(db, f.id) == []
    db.refresh(f)
    assert f.verified is True


def test_a_box_with_no_classification_method_is_discarded_too(client, db):
    """`classification_method` is nullable, and in SQL `!= 'human'` matches
    no NULLs. A bare inequality would leave every unclassified box behind,
    which is most of them."""
    f = _file_in_project(db)
    make_detection(db, file_id=f.id, confidence=0.05, classification_method=None)
    make_detection(db, file_id=f.id, confidence=0.05, classification_method="ai")
    db.commit()

    _verify(client, f.id)

    assert _boxes(db, f.id) == []


def test_unverifying_does_not_bring_the_boxes_back(client, db):
    f = _file_in_project(db)
    make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    _verify(client, f.id)
    _verify(client, f.id, verified=False)

    assert _boxes(db, f.id) == []
    db.refresh(f)
    assert f.verified is False


def test_verifying_a_file_with_a_passing_box_deletes_the_weak_ones(client, db):
    """Same rule on a file that is not empty: the box the person saw is
    signed off, the one they could not see is gone. Keeping it left a
    verified file that grew a box the moment the slider dropped."""
    f = _file_in_project(db)
    strong = make_detection(db, file_id=f.id, confidence=0.9)
    make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    _verify(client, f.id)

    assert {d.id for d in _boxes(db, f.id)} == {strong.id}
    db.refresh(strong)
    assert strong.verified is True
    db.refresh(f)
    assert f.verified is True


def test_the_verify_write_is_frame_gated(client, db):
    """A video is judged on its best frame. Boxes on the frames nobody saw
    are neither deleted nor signed off, strong or weak. Before this the
    verify write had no frame clause and signed off boxes on frames the
    person never opened."""
    f = _file_in_project(db, file_type="video", best_frame_number=0)
    on_weak = make_detection(db, file_id=f.id, confidence=0.05, frame_number=0).id
    on_strong = make_detection(db, file_id=f.id, confidence=0.9, frame_number=0).id
    off_weak = make_detection(db, file_id=f.id, confidence=0.05, frame_number=7).id
    off_strong = make_detection(db, file_id=f.id, confidence=0.9, frame_number=7).id
    db.commit()

    _verify(client, f.id)

    surviving = {d.id: d for d in _boxes(db, f.id)}
    assert on_weak not in surviving
    assert surviving[on_strong].verified is True
    assert surviving[off_weak].verified is False
    assert surviving[off_strong].verified is False


def test_verifying_a_file_recomputes_what_it_is_about(client, db):
    """Deleting boxes and verifying boxes both move the strongest passing
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


def test_only_boxes_on_the_frame_the_person_saw_are_discarded(client, db):
    """A clip reads empty on its best frame alone, and that frame is the
    only one the app shows. Boxes on the other sampled frames were never
    on screen, so a verdict about the visible frame is not a verdict
    about them. Without the visible-frame scope, one Enter on a video
    threw away every box in the clip, including a confident animal on a
    frame nobody had opened."""
    f = _file_in_project(db, file_type="video", best_frame_number=0)
    on_screen = make_detection(
        db, file_id=f.id, confidence=0.05, frame_number=0
    ).id
    off_screen = make_detection(
        db, file_id=f.id, confidence=0.95, frame_number=7
    ).id
    db.commit()

    _verify(client, f.id)

    surviving = {d.id for d in _boxes(db, f.id)}
    assert on_screen not in surviving
    assert off_screen in surviving


def test_a_drawn_box_survives_a_file_verify(client, db):
    """The rule never reads who drew a box, and it does not have to: a
    drawn box is verified at confidence 1.0, so it is always visible and
    never below the threshold. Here it sits off the best frame, where
    nothing is touched at all, so the noise beside it survives too.
    """
    f = _file_in_project(db, file_type="video", best_frame_number=0)
    drawn_id = make_detection(
        db,
        file_id=f.id,
        confidence=1.0,
        frame_number=5,
        verified=True,
        classification_method="human",
    ).id
    noise_id = make_detection(
        db, file_id=f.id, confidence=0.05, frame_number=5
    ).id
    db.commit()

    _verify(client, f.id)

    surviving = {d.id for d in _boxes(db, f.id)}
    assert surviving == {drawn_id, noise_id}


def test_a_rejected_box_does_not_make_a_file_look_occupied(client, db):
    """A file whose only real box was marked false reads as empty in the
    Files tab (`is_a_real_detection()`), and verifying it deletes the weak
    boxes beside the rejected one. The rejected box itself is kept: a
    person looked at it and judged it, and it is verified, so the rule
    never touches it.

    The untick in the middle mirrors the user's route: marking a box
    false verifies it, which rolls up and leaves the file signed off, so
    a person unticks it to look again and then signs it off as empty.
    """
    f = _file_in_project(db)
    rejected_id = make_detection(db, file_id=f.id, confidence=0.85).id
    noise_id = make_detection(db, file_id=f.id, confidence=0.05).id
    db.commit()

    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": [rejected_id], "label": "false detection"},
    )
    assert resp.status_code == 200, resp.text

    project_id = f.deployment.project_id
    empties = client.get(
        f"/api/projects/{project_id}/labels/files", params={"empty": "show_only"}
    ).json()
    assert empties["total"] == 1, "the file reads as empty"

    _verify(client, f.id, verified=False)
    _verify(client, f.id)

    surviving = {d.id for d in _boxes(db, f.id)}
    assert noise_id not in surviving, (
        "the weak box survived a verdict of 'nothing here'"
    )
    assert surviving == {rejected_id}
