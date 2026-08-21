"""The fields the drawing rule reads must actually reach the wire.

`shouldDrawBbox` in `frontend/src/lib/detection-utils.ts` decides which
boxes paint over a photo, and it reads two things off each detection in
`GET /api/files/{id}`:

* `verified` — a box a human confirmed draws at any confidence, the
  same `confidence >= threshold OR verified` rule every backend query
  applies.
* `job_id` — null exactly for a box a person drew, which is what the
  empties viewer shows and the detector's boxes are not.

Neither was in `schemas/file.py`'s `DetectionResponse`, while the
frontend's `DetectionResponse` in `api/types.ts` declared both. That
combination is invisible to both type checkers: TypeScript believes its
own declaration, and Python never sees it. Reading either field gave
`undefined`, which reads as "not verified" and as "not human-drawn", so
the verified override would have silently never fired and the empties
viewer would have drawn nothing at all.

A frontend unit test cannot catch this either, because it would build
its fixture from the same lying type. The wire is the only place the
two sides meet, so the assertion belongs here.
"""

from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_job,
    make_project,
    make_site,
)


def _file_with_detections(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    return make_file(db, deployment_id=deployment.id)


def test_file_detections_carry_verified_and_job_id(client, db):
    """Both fields are present and typed, not merely absent-and-defaulted."""
    f = _file_with_detections(db)
    job = make_job(db)
    make_detection(db, file_id=f.id, confidence=0.9, job_id=job.id)
    db.commit()

    resp = client.get(f"/api/files/{f.id}")
    assert resp.status_code == 200
    det = resp.json()["detections"][0]

    assert "verified" in det, "shouldDrawBbox reads this; without it every box reads unverified"
    assert "job_id" in det, "isHumanDrawnBox reads this; without it no box reads human-drawn"
    assert det["verified"] is False
    assert det["job_id"] == job.id


def test_a_human_drawn_box_reports_a_null_job_id(client, db):
    """`job_id is None` is the marker the empties viewer draws by.

    It has to survive serialization as JSON `null` rather than being
    dropped, since a missing key and a null key are the same
    `undefined` in the browser but only one of them is the truth.
    """
    f = _file_with_detections(db)
    make_detection(
        db,
        file_id=f.id,
        confidence=1.0,
        job_id=None,
        verified=True,
        classification_method="human",
    )
    db.commit()

    det = client.get(f"/api/files/{f.id}").json()["detections"][0]
    assert det["job_id"] is None
    assert det["verified"] is True


def test_a_verified_low_confidence_box_still_reports_verified(client, db):
    """Relabelling never rewrites `Detection.confidence`.

    So a box a human confirmed keeps whatever the detector scored it,
    and `verified` is the only thing on the payload that says it should
    still be drawn. Pinning the pair together is the point: a fixture
    that verified a 0.9 box would pass while the case that matters
    stayed broken.
    """
    f = _file_with_detections(db)
    make_detection(db, file_id=f.id, confidence=0.03, verified=True)
    db.commit()

    det = client.get(f"/api/files/{f.id}").json()["detections"][0]
    assert det["confidence"] == 0.03
    assert det["verified"] is True
