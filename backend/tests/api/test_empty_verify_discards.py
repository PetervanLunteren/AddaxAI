"""Verifying an empty file throws the detector's boxes away.

Saying a frame is empty is a statement about the photograph: there is no
animal in it. Every box the detector left on it is therefore wrong, so
they are deleted rather than kept below the threshold.

Keeping them made "empty" true only at the threshold it was checked at.
Drop the confidence slider and the file came back carrying a 3% smudge
while still flagged verified; raise the project threshold afterwards and
it exported ``is_verified = TRUE`` beside a species nobody had confirmed.

This is only defensible because the empties viewer draws no boxes over
the frame, so the person is judging the picture and not a threshold. If
weak boxes are ever drawn there again, these tests should be revisited
along with the rule.
"""

from app.models import Detection
from tests.conftest import (
    make_deployment,
    make_detection,
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


def test_a_file_with_a_passing_box_keeps_everything(client, db):
    """The other branch. Nothing is empty here, so verifying signs the
    boxes off instead of deleting them, including the weak ones that ride
    along untouched."""
    f = _file_in_project(db)
    strong = make_detection(db, file_id=f.id, confidence=0.9)
    weak = make_detection(db, file_id=f.id, confidence=0.05)
    db.commit()

    _verify(client, f.id)

    kept = {d.id for d in _boxes(db, f.id)}
    assert kept == {strong.id, weak.id}
    db.refresh(strong)
    db.refresh(weak)
    assert strong.verified is True
    # Below the floor, so invisible to the user and never theirs to sign.
    assert weak.verified is False


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


def test_a_file_holding_a_drawn_box_is_never_treated_as_empty(client, db):
    """The reason the delete cannot eat a person's own box.

    `on_visible_frame_of` passes verified detections on *any* frame, so a
    drawn box keeps its file reviewable even on a video where it sits off
    the best frame. That makes the empty branch unreachable while one
    exists, which is what the belt-and-braces clause in
    `_discard_detector_boxes` is insurance against rather than a live
    path. If this test ever fails, that clause is the thing holding the
    line and it must stay.
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
