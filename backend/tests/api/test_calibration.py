"""
Tests for the Calibration page backend (`crud/calibration.py`, `routers/calibration.py`).

1. Which detections the sample draws from (same scope as the Labels grid).
2. Answers: stored with the AI's label and score of that moment, never
   touching verification.
3. The per-label summary: counts, the minimum, the rule-of-three
   fallback, thresholds and curve.
"""

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.api.crud import calibration as calibration_crud
from app.models import CalibrationAnswer, Detection
from tests.conftest import make_deployment, make_detection, make_file, make_project, make_site


def _project(db: Session, threshold: float = 0.5):  # type: ignore[no-untyped-def]
    project = make_project(db, classification_model_id="CLS-TEST-v1", counting_threshold=threshold)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    return project, deployment


def _wolf(db: Session, deployment_id: str, score: float, **kw) -> Detection:  # type: ignore[no-untyped-def]
    f = make_file(db, deployment_id=deployment_id)
    defaults = dict(
        label="wolf",
        label_confidence=score,
        original_label="wolf",
        original_label_confidence=score,
    )
    defaults.update(kw)
    return make_detection(db, file_id=f.id, **defaults)


def test_sample_scope_matches_the_labels_grid(db: Session) -> None:
    project, dep = _project(db, threshold=0.5)
    shown = _wolf(db, dep.id, 0.8)
    relabelled = _wolf(db, dep.id, 0.7, label="dog", verified=True)
    _wolf(db, dep.id, 0.9, confidence=0.3)  # below the counting threshold
    _wolf(db, dep.id, 0.9, original_label="blank")  # a "nothing here" label
    _wolf(db, dep.id, 0.9, original_label=None, original_label_confidence=None)
    video = make_file(db, deployment_id=dep.id, file_type="video", best_frame_number=3)
    make_detection(
        db, file_id=video.id, frame_number=5, label="wolf", original_label="wolf",
        original_label_confidence=0.9,
    )  # a frame nobody can open
    db.commit()

    ids = {item.detection_id for item in calibration_crud.draw_batch(db, project.id, "wolf")}
    assert ids == {shown.id, relabelled.id}
    row = calibration_crud.get_calibration(db, project.id).labels[0]
    assert (row.label, row.detections) == ("wolf", 2)


def test_sample_skips_answered_detections_and_caps_the_batch(db: Session) -> None:
    project, dep = _project(db)
    dets = [_wolf(db, dep.id, 0.5 + i / 100) for i in range(30)]
    db.commit()
    calibration_crud.save_answer(db, dets[0].id, "wolf", "right")
    calibration_crud.save_answer(db, dets[1].id, "wolf", "unsure")

    batch = calibration_crud.draw_batch(db, project.id, "wolf", rng=np.random.default_rng(1))
    ids = [item.detection_id for item in batch]
    assert len(ids) == calibration_crud.BATCH_SIZE
    assert dets[0].id not in ids and dets[1].id not in ids


def test_batch_reads_a_bounded_slice_and_keeps_the_spread(db: Session, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """One dominant label: Python sees at most N_BINS x BATCH_SIZE rows,
    and the few low scores still make it into the batch."""
    project, dep = _project(db)
    for _ in range(300):
        _wolf(db, dep.id, 0.999)
    low = {_wolf(db, dep.id, s).id for s in (0.1, 0.3, 0.5, 0.7)}
    db.commit()

    seen: list[int] = []
    real = calibration_crud.draw_sample

    def spy(candidates, **kw):  # type: ignore[no-untyped-def]
        seen.append(len(candidates))
        return real(candidates, **kw)

    monkeypatch.setattr(calibration_crud, "draw_sample", spy)
    ids = {i.detection_id for i in calibration_crud.draw_batch(db, project.id, "wolf")}
    assert seen[0] <= calibration_crud.N_BINS * calibration_crud.BATCH_SIZE
    assert len(ids) == calibration_crud.BATCH_SIZE
    assert low <= ids


def test_answer_keeps_the_ai_call_and_leaves_verification_alone(db: Session) -> None:
    project, dep = _project(db)
    det = _wolf(db, dep.id, 0.83)
    db.commit()

    calibration_crud.save_answer(db, det.id, "wolf", "wrong")
    # A reprocess with other settings changes the AI's call afterwards.
    det.original_label = "dog"
    det.original_label_confidence = 0.99
    db.commit()

    stored = db.query(CalibrationAnswer).one()
    assert (stored.label, stored.score, stored.answer) == ("wolf", 0.83, "wrong")
    db.refresh(det)
    assert det.verified is False and det.label == "wolf"


def test_answering_again_replaces_the_answer(db: Session) -> None:
    project, dep = _project(db)
    det = _wolf(db, dep.id, 0.6)
    db.commit()
    calibration_crud.save_answer(db, det.id, "wolf", "wrong")
    calibration_crud.save_answer(db, det.id, "wolf", "right")
    assert [a.answer for a in db.query(CalibrationAnswer).all()] == ["right"]


def test_answer_is_removed_with_its_detection(db: Session) -> None:
    project, dep = _project(db)
    det = _wolf(db, dep.id, 0.6)
    db.commit()
    calibration_crud.save_answer(db, det.id, "wolf", "right")
    db.delete(det)
    db.commit()
    assert db.query(CalibrationAnswer).count() == 0


def test_below_the_minimum_shows_what_is_still_needed(db: Session) -> None:
    project, dep = _project(db)
    dets = [_wolf(db, dep.id, 0.5 + i / 100) for i in range(8)]
    db.commit()
    for d in dets[:6]:
        calibration_crud.save_answer(db, d.id, "wolf", "right")
    calibration_crud.save_answer(db, dets[6].id, "wolf", "wrong")
    calibration_crud.save_answer(db, dets[7].id, "wolf", "unsure")

    row = calibration_crud.get_calibration(db, project.id).labels[0]
    assert (row.right, row.wrong, row.unsure) == (6, 1, 1)
    assert (row.right_needed, row.wrong_needed) == (4, 9)
    assert row.curve is None and row.thresholds is None and row.no_error_bound is None


def test_no_errors_gives_the_rule_of_three_bound(db: Session) -> None:
    project, dep = _project(db)
    dets = [_wolf(db, dep.id, 0.9) for _ in range(60)]
    db.commit()
    for d in dets:
        calibration_crud.save_answer(db, d.id, "wolf", "right")

    row = calibration_crud.get_calibration(db, project.id).labels[0]
    assert row.no_error_bound is not None and abs(row.no_error_bound - 0.95) < 1e-9
    assert row.curve is None


def test_enough_answers_give_thresholds_and_a_curve(db: Session) -> None:
    project, dep = _project(db)
    rng = np.random.default_rng(3)
    for _ in range(60):
        score = float(np.round(rng.uniform(0.5, 0.999), 3))
        det = _wolf(db, dep.id, score)
        db.commit()
        p = 1 / (1 + np.exp(-(2 + 1.5 * np.log(score / (1 - score)))))
        calibration_crud.save_answer(db, det.id, "wolf", "right" if rng.uniform() < p else "wrong")
    # Keep the minimum reachable whatever the draw.
    for score in (0.5, 0.52, 0.55, 0.57, 0.6, 0.62, 0.65, 0.67, 0.7, 0.72):
        det = _wolf(db, dep.id, score)
        db.commit()
        calibration_crud.save_answer(db, det.id, "wolf", "wrong")

    row = calibration_crud.get_calibration(db, project.id).labels[0]
    assert row.thresholds is not None and row.curve is not None
    assert [t.target for t in row.thresholds] == [0.9, 0.95, 0.99]
    assert row.curve[0].score == 0.5
    assert all(0 <= pt.lower <= pt.p <= pt.upper <= 1 for pt in row.curve)


def test_endpoints(client, db: Session) -> None:  # type: ignore[no-untyped-def]
    project, dep = _project(db)
    det = _wolf(db, dep.id, 0.7)
    db.commit()

    sample = client.get(f"/api/calibration/sample?project_id={project.id}&label=wolf")
    assert sample.status_code == 200
    assert sample.json() == [{"detection_id": det.id, "score": 0.7}]

    def post(detection_id: str, answer: str, label: str = "wolf") -> int:
        body = {"detection_id": detection_id, "label": label, "answer": answer}
        return int(client.post("/api/calibration/answers", json=body).status_code)

    assert post(det.id, "right") == 204
    assert post(det.id, "maybe") == 422
    assert post("nope", "right") == 404
    assert post(det.id, "right", label="dog") == 404

    overview = client.get(f"/api/calibration?project_id={project.id}")
    assert overview.status_code == 200
    assert overview.json()["labels"][0]["right"] == 1
    assert client.get("/api/calibration?project_id=nope").status_code == 404


def test_answer_outside_the_sample_is_refused(db: Session) -> None:
    project, dep = _project(db, threshold=0.5)
    hidden = _wolf(db, dep.id, 0.9, confidence=0.3)  # below the counting threshold
    blank = _wolf(db, dep.id, 0.9, original_label="blank")
    db.commit()
    for det in (hidden, blank):
        with pytest.raises(ValueError):
            calibration_crud.save_answer(db, det.id, det.original_label, "right")
    assert db.query(CalibrationAnswer).count() == 0


def test_answer_after_a_reprocess_relabelled_it_is_refused(db: Session) -> None:
    """A reprocess between drawing the batch and answering rewrites
    `original_label`: the answer was about the old label, so it must not
    land under the new one."""
    project, dep = _project(db)
    det = _wolf(db, dep.id, 0.8)
    db.commit()
    assert [i.detection_id for i in calibration_crud.draw_batch(db, project.id, "wolf")] == [det.id]
    det.original_label = "dog"
    db.commit()
    with pytest.raises(ValueError):
        calibration_crud.save_answer(db, det.id, "wolf", "right")
    assert db.query(CalibrationAnswer).count() == 0


def test_a_label_that_will_not_fit_keeps_the_other_rows(db: Session, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    project, dep = _project(db)
    for label in ("wolf", "lynx"):
        for i in range(20):
            det = _wolf(db, dep.id, 0.5 + i / 50, original_label=label)
            db.commit()
            calibration_crud.save_answer(db, det.id, label, "right" if i >= 10 else "wrong")

    real_fit = calibration_crud.fit_calibration

    def flaky_fit(scores, right):  # type: ignore[no-untyped-def]
        if len(scores) == 20 and flaky_fit.calls == 0:
            flaky_fit.calls += 1
            raise RuntimeError("did not converge")
        return real_fit(scores, right)

    flaky_fit.calls = 0  # type: ignore[attr-defined]
    monkeypatch.setattr(calibration_crud, "fit_calibration", flaky_fit)
    rows = calibration_crud.get_calibration(db, project.id).labels
    assert len(rows) == 2
    assert sum(r.curve is None for r in rows) == 1
    assert sum(r.curve is not None for r in rows) == 1
