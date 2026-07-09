"""Observation -> file verification cascade.

`File.verified` is a maintained rollup: a file is verified when every
reviewable detection on it is verified. Verifying detections (the
observation level) cascades up to flip `File.verified`; the file-verify
action cascades back down to the detections.
"""

from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
)


def _file_with_two_detections(db, threshold=0.5):
    project = make_project(db, counting_threshold=threshold)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id, project_id=project.id)
    f = make_file(db, deployment_id=dep.id, verified=False)
    d1 = make_detection(db, file_id=f.id, confidence=0.9, verified=False)
    d2 = make_detection(db, file_id=f.id, confidence=0.9, verified=False)
    db.commit()
    return project, f, d1, d2


def test_verifying_all_detections_marks_file_verified(client, db):
    _, f, d1, d2 = _file_with_two_detections(db)
    assert f.verified is False

    # Verify one of two -> file still unverified.
    client.post("/api/detections/bulk-verify", json={"detection_ids": [d1.id]})
    db.refresh(f)
    assert f.verified is False

    # Verify the second -> all reviewable verified -> file flips verified.
    client.post("/api/detections/bulk-verify", json={"detection_ids": [d2.id]})
    db.refresh(f)
    assert f.verified is True


def test_unverifying_a_detection_unflips_file(client, db):
    _, f, d1, d2 = _file_with_two_detections(db)
    client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [d1.id, d2.id]},
    )
    db.refresh(f)
    assert f.verified is True

    client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [d1.id], "verified": False},
    )
    db.refresh(f)
    assert f.verified is False


def test_below_threshold_detections_do_not_block_verification(client, db):
    project, f, d1, d2 = _file_with_two_detections(db, threshold=0.5)
    # A below-threshold, unverified detection is not reviewable, so it
    # must not keep the file unverified once the reviewable ones are done.
    make_detection(db, file_id=f.id, confidence=0.2, verified=False)
    db.commit()

    client.post(
        "/api/detections/bulk-verify",
        json={"detection_ids": [d1.id, d2.id]},
    )
    db.refresh(f)
    assert f.verified is True


def test_file_verify_action_cascades_down(client, db):
    _, f, d1, d2 = _file_with_two_detections(db)
    # The file-verify action (file PATCH verified=True) sets File.verified
    # and cascades down to the detections.
    client.patch(f"/api/files/{f.id}", json={"verified": True})
    db.refresh(f)
    db.refresh(d1)
    db.refresh(d2)
    assert f.verified is True
    assert d1.verified is True
    assert d2.verified is True
