"""The `File.verified` rollup must read the changes its caller just made.

The app's session runs with ``autoflush=False`` (``db/base.py``). A
caller that sets ``det.verified = True`` in Python therefore still has it
pending, and a query issued before a flush reads the *old* value from the
database. `recompute_file_verified` then decided "not all verified" and
left the flag alone, because the value it computed matched the one
already stored.

The symptom was quiet and reached exported data: relabelling a file, or
marking its detection false, left ``File.verified`` at FALSE, so
``addaxai-files.csv`` reported ``is_verified = FALSE`` for a file the
person had just judged. Relabelling the same detection a second time
fixed it, because by then the first write had landed.

**These tests build their own session with autoflush off.** The shared
test session in ``conftest`` uses SQLAlchemy's default, ``autoflush=True``,
which is why the existing suite never caught this: under that setting the
pending change is written before the query and the rollup is correct. Any
test using the shared session would pass whether or not the fix is
present, which makes it worthless as a guard. Aligning the whole suite is
worth doing on its own, and it surfaces three further failures of the
same family in the event and export paths.
"""

from sqlalchemy.orm import sessionmaker

from app.api.routers.detections import (
    BulkRelabelRequest,
    bulk_relabel_detections,
)
from app.models import Detection, File
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
)


def _production_like_session(db):
    """A session bound to the test engine but with the app's settings."""
    return sessionmaker(bind=db.get_bind(), autoflush=False)()


def _one_file(db):
    project = make_project(db, counting_threshold=0.2)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=deployment.id)
    d = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
    db.commit()
    return f.id, d.id


def test_relabelling_rolls_the_file_up_on_the_first_try(db):
    file_id, det_id = _one_file(db)

    prod = _production_like_session(db)
    try:
        bulk_relabel_detections(
            BulkRelabelRequest(detection_ids=[det_id], label="chital"), prod
        )
    finally:
        prod.close()

    db.expire_all()
    assert db.get(Detection, det_id).verified is True
    assert db.get(File, file_id).verified is True


def test_marking_false_rolls_the_file_up_too(db):
    """Same path, and the one that reaches exports as `is_verified`."""
    file_id, det_id = _one_file(db)

    prod = _production_like_session(db)
    try:
        bulk_relabel_detections(
            BulkRelabelRequest(
                detection_ids=[det_id], label="false detection"
            ),
            prod,
        )
    finally:
        prod.close()

    db.expire_all()
    assert db.get(File, file_id).verified is True
    # And the file is blank, because a rejected box cannot be its subject.
    assert db.get(File, file_id).observation_type == "blank"
