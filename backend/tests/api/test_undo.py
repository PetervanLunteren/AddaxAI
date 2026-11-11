"""Undo puts back exactly what was there before the last action.

Every undoable action returns the before-state of the rows it changes
(``before``), and ``POST /api/undo`` writes it back. The old undo reverted
to the model's prediction instead, which lost earlier human corrections
and could not re-verify anything. These tests pin the cases that used
to lose data.
"""

from datetime import datetime

from sqlalchemy import select

from app.ml.taxonomy_db import ensure_builtin_labels, resolve_taxonomy_id
from app.models import Detection, EventObservation, File
from app.models.event import event_files
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
    make_track,
)


def _project_file(db, threshold=0.2):
    p = make_project(db, counting_threshold=threshold)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    return p, d, make_file(db, deployment_id=d.id)


def _relabel(client, ids, label, **extra):
    resp = client.post(
        "/api/detections/bulk-relabel",
        json={"detection_ids": ids, "label": label, **extra},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["before"]


def _verify_files(client, ids, verified=True):
    resp = client.post(
        "/api/files/bulk-verify", json={"file_ids": ids, "verified": verified}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["before"]


def _undo(client, before):
    resp = client.post("/api/undo", json=before)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_undo_puts_back_the_earlier_human_label_not_the_machines(client, db):
    """Relabel twice, undo twice: the first undo lands on the first human
    label, the second on the machine's. The old revert skipped straight
    to the machine and lost the human step."""
    p, _, f = _project_file(db)
    det = make_detection(
        db,
        file_id=f.id,
        label="deer",
        label_confidence=0.42,
        original_label="deer",
        original_label_confidence=0.42,
        classification_method="machine",
    )
    db.commit()

    first = _relabel(client, [det.id], "elk")
    db.expire_all()
    elk_at = det.verified_at_utc
    assert det.label == "elk" and det.verified is True
    second = _relabel(client, [det.id], "moose")
    db.expire_all()
    assert det.label == "moose"

    _undo(client, second)
    db.expire_all()
    assert det.label == "elk"
    assert det.label_confidence == 1.0
    assert det.classification_method == "human"
    assert det.verified is True
    assert det.verified_at_utc == elk_at
    assert det.label_taxonomy_id == resolve_taxonomy_id("elk", p.id, db)

    _undo(client, first)
    db.expire_all()
    assert det.label == "deer"
    assert det.label_confidence == 0.42
    assert det.classification_method == "machine"
    assert det.verified is False
    assert det.verified_at_utc is None
    assert det.original_label == "deer"


def test_undoing_a_file_sign_off_brings_the_rejected_weak_box_back(client, db):
    """Verifying a file rejects its weak boxes as false detections. The
    snapshot includes them, so undo gives them their label back."""
    p, _, f = _project_file(db, threshold=0.2)
    strong = make_detection(
        db, file_id=f.id, confidence=0.9, label="fox",
        original_label="fox", classification_method="machine",
    )
    weak = make_detection(
        db, file_id=f.id, confidence=0.05, label="deer",
        original_label="deer", classification_method="machine",
    )
    db.commit()

    before = _verify_files(client, [f.id])
    db.expire_all()
    assert weak.label == "false detection" and weak.verified is True
    assert f.verified is True
    assert {s["detection_id"] for s in before["detections"]} == {strong.id, weak.id}
    assert [s["file_id"] for s in before["files"]] == [f.id]

    _undo(client, before)
    db.expire_all()
    assert weak.label == "deer"
    assert weak.verified is False
    assert weak.classification_method == "machine"
    assert weak.original_label == "deer"
    assert weak.label_taxonomy_id == resolve_taxonomy_id("deer", p.id, db)
    assert strong.label == "fox" and strong.verified is False
    assert f.verified is False and f.verified_at_utc is None


def test_undoing_an_unverify_puts_the_sign_off_back_with_its_timestamps(client, db):
    _, _, f = _project_file(db)
    det = make_detection(db, file_id=f.id, confidence=0.9, label="fox")
    db.commit()
    _verify_files(client, [f.id])
    db.expire_all()
    file_at, det_at = f.verified_at_utc, det.verified_at_utc
    assert file_at is not None and det_at is not None

    before = _verify_files(client, [f.id], verified=False)
    db.expire_all()
    assert f.verified is False and det.verified is False

    _undo(client, before)
    db.expire_all()
    assert f.verified is True and f.verified_at_utc == file_at
    assert det.verified is True and det.verified_at_utc == det_at


def test_undoing_a_box_verify_puts_the_files_rollup_back(client, db):
    """Verifying the last box flips the file through the rollup. The
    snapshot carries the file too, so undo flips it back."""
    _, _, f = _project_file(db)
    det = make_detection(db, file_id=f.id, confidence=0.9, label="fox")
    db.commit()

    resp = client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [det.id], "verified": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["updated_count"] == 1
    assert body["before"]["files"] == [
        {"file_id": f.id, "verified": False, "verified_at_utc": None}
    ]
    db.expire_all()
    assert f.verified is True

    _undo(client, body["before"])
    db.expire_all()
    assert det.verified is False
    assert f.verified is False


def test_undo_rederives_counts_and_observation_type_past_the_request(client, db):
    """What hangs off ``verified`` is re-derived on restore, and the
    result survives the end of the request (``get_db`` never commits)."""
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

    resp = client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [weak.id], "verified": True},
    )
    assert resp.status_code == 200, resp.text
    db.expire_all()
    assert db.get(File, file_id).observation_type == "animal"
    assert db.query(EventObservation).filter_by(event_id=ev.id).count() == 1

    _undo(client, resp.json()["before"])
    # Discards exactly what close() discards; whatever is left was committed.
    db.rollback()
    assert db.get(Detection, weak.id).verified is False
    assert db.get(File, file_id).observation_type == "blank"
    assert db.query(EventObservation).filter_by(event_id=ev.id).count() == 0


def test_undo_to_no_label_lands_on_the_builtin_animal_row(client, db):
    """A box that had no species goes back to its category's builtin
    row, not to no row at all: the label filter matches on that id."""
    animal_row = ensure_builtin_labels(db)["animal"]
    _, _, f = _project_file(db)
    det = make_detection(db, file_id=f.id, label=None)
    db.commit()

    before = _relabel(client, [det.id], "deer")
    db.expire_all()
    assert det.label == "deer"

    body = _undo(client, before)
    assert body["detections"][0]["label"] is None
    db.expire_all()
    assert det.label is None
    assert det.label_taxonomy_id == animal_row
    assert det.common_name == "Animal"
    assert det.verified is False


def test_undo_puts_a_changed_category_back(client, db):
    _, _, f = _project_file(db)
    det = make_detection(db, file_id=f.id, category="animal", label="fox")
    db.commit()

    before = _relabel(client, [det.id], None, category="vehicle")
    db.expire_all()
    assert det.category == "vehicle" and det.label is None

    _undo(client, before)
    db.expire_all()
    assert det.category == "animal"
    assert det.label == "fox"


def test_undo_of_nothing_is_a_no_op_and_of_ghosts_a_404(client):
    assert client.post("/api/undo", json={}).json() == {"detections": []}
    ghost = {
        "detections": [
            {
                "detection_id": "ghost",
                "category": "animal",
                "label": None,
                "label_confidence": None,
                "scientific_name": None,
                "common_name": None,
                "classification_method": None,
                "verified": False,
                "verified_at_utc": None,
            }
        ],
        "files": [],
    }
    assert client.post("/api/undo", json=ghost).status_code == 404


def test_undo_skips_rows_that_are_gone_and_restores_the_rest(client, db):
    _, _, f = _project_file(db)
    a = make_detection(db, file_id=f.id, label="fox")
    b = make_detection(db, file_id=f.id, label="fox")
    db.commit()

    before = _relabel(client, [a.id, b.id], "elk")
    db.delete(db.get(Detection, b.id))
    db.commit()

    body = _undo(client, before)
    assert [r["id"] for r in body["detections"]] == [a.id]
    db.expire_all()
    assert a.label == "fox" and a.verified is False


def test_undoing_a_clip_sign_off_hands_the_track_its_own_labels_back(client, db):
    """Signing a clip off gives each track its card's label on every frame.
    The snapshot holds every box of the file, so undo puts back the frame
    the classifier read differently, not only the card."""
    p, d, _ = _project_file(db, threshold=0.2)
    clip = make_file(db, deployment_id=d.id, file_type="video", file_format="mp4")
    track = make_track(db, file_id=clip.id, start_frame=0, end_frame=60,
                       representative_frame_number=30)
    boxes = {
        frame: make_detection(
            db, file_id=clip.id, frame_number=frame, track_id=track.id,
            confidence=0.9, label=label, original_label=label,
            classification_method="machine",
        )
        for frame, label in ((0, "fox"), (30, "fox"), (60, "badger"))
    }
    db.commit()

    before = _verify_files(client, [clip.id])
    db.expire_all()
    assert boxes[60].label == "fox"
    assert {s["detection_id"] for s in before["detections"]} == {b.id for b in boxes.values()}

    _undo(client, before)
    db.expire_all()
    assert boxes[60].label == "badger" and boxes[60].classification_method == "machine"
    assert not any(b.verified for b in boxes.values())
    assert clip.verified is False


def test_undo_restores_more_boxes_than_sqlite_takes_parameters(client, db):
    """A track-wide snapshot of a long clip holds thousands of boxes. Older
    SQLite builds allow 999 bound parameters, so restore reads in chunks."""
    import sqlite3

    _, d, _ = _project_file(db)
    f = make_file(db, deployment_id=d.id, file_type="video", file_format="mp4")
    track = make_track(db, file_id=f.id, start_frame=0, end_frame=1199,
                       representative_frame_number=0, frame_count=1200)
    card = None
    for frame in range(1200):
        box = make_detection(db, file_id=f.id, frame_number=frame, track_id=track.id,
                             confidence=0.9, label="fish")
        card = card or box
    db.commit()
    # One card: its label reaches all 1,200 boxes, and so does the snapshot.
    before = _relabel(client, [card.id], "shark")
    assert len(before["detections"]) == 1200

    raw = db.connection().connection.driver_connection
    limit = raw.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    raw.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        _undo(client, before)
    finally:
        raw.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    db.expire_all()
    assert {det.label for det in db.query(Detection).filter(Detection.file_id == f.id)} == {
        "fish"
    }


def test_undo_looks_a_label_up_once_not_once_per_box(client, db, monkeypatch):
    """A track's boxes share their label, and a long clip's snapshot holds
    tens of thousands of them. One lookup per box made undoing a 12,000-box
    verdict take five times as long as the verdict itself."""
    from app.api.crud import undo as undo_crud

    p, d, _ = _project_file(db)
    clip = make_file(db, deployment_id=d.id, file_type="video", file_format="mp4")
    track = make_track(db, file_id=clip.id, start_frame=0, end_frame=40,
                       representative_frame_number=0, frame_count=5)
    boxes = [
        make_detection(db, file_id=clip.id, frame_number=frame, track_id=track.id,
                       confidence=0.9, label="fish")
        for frame in range(0, 50, 10)
    ]
    db.commit()
    before = _relabel(client, [boxes[0].id], "shark")
    assert len(before["detections"]) == 5

    lookups = []
    resolve = undo_crud._resolve_detection_taxonomy

    def counted(db, det, label):
        lookups.append(label)
        return resolve(db, det, label)

    monkeypatch.setattr(undo_crud, "_resolve_detection_taxonomy", counted)
    _undo(client, before)

    assert lookups == ["fish"]
    db.expire_all()
    fish_id = resolve_taxonomy_id("fish", p.id, db)
    assert fish_id is not None
    assert {(b.label, b.label_taxonomy_id) for b in boxes} == {("fish", fish_id)}

