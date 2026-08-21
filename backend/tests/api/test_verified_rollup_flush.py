"""The `File.verified` rollup must read the changes its caller just made.

The session runs with ``autoflush=False`` (``db/base.py``, mirrored by
``tests/conftest.py``). A caller that sets ``det.verified = True`` in
Python therefore still has it pending, and a query issued before a flush
reads the *old* value from the database. `recompute_file_verified` then
decided "not all verified" and left the flag alone, because the value it
computed matched the one already stored.

The symptom was quiet and reached exported data: relabelling a file, or
marking its detection false, left ``File.verified`` at FALSE, so
``addaxai-files.csv`` reported ``is_verified = FALSE`` for a file the
person had just judged. Relabelling the same detection a second time
fixed it, because by then the first write had landed.

These tests used to build their own ``autoflush=False`` session, because
the shared one defaulted to autoflush *on* and would have passed whether
or not the fix was present. The shared session now carries the app's
setting, so they use it, and `test_the_test_session_keeps_the_apps_flush_semantics`
is what stops that drifting apart again.
"""

from app.api.routers.detections import (
    BulkRelabelRequest,
    bulk_relabel_detections,
)
from app.db.base import get_session_factory
from app.models import Detection, File
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
    make_site,
)


def _one_file(db):
    project = make_project(db, counting_threshold=0.2)
    site = make_site(db, project_id=project.id)
    deployment = make_deployment(db, site_id=site.id)
    f = make_file(db, deployment_id=deployment.id)
    d = make_detection(db, file_id=f.id, category="animal", confidence=0.9)
    db.commit()
    return f.id, d.id


def test_the_test_session_keeps_the_apps_flush_semantics(db):
    """The suite must read the way the app reads.

    Both halves are asserted, so flipping either one goes red: the app
    turning autoflush back on, or the test session drifting back to
    SQLAlchemy's default. Aligning them is what lets every other test in
    the suite catch a stale read instead of hiding it.
    """
    assert db.autoflush is False
    assert get_session_factory().kw["autoflush"] is False


def test_relabelling_rolls_the_file_up_on_the_first_try(db):
    file_id, det_id = _one_file(db)

    bulk_relabel_detections(
        BulkRelabelRequest(detection_ids=[det_id], label="chital"), db
    )

    db.expire_all()
    assert db.get(Detection, det_id).verified is True
    assert db.get(File, file_id).verified is True


def test_marking_false_rolls_the_file_up_too(db):
    """Same path, and the one that reaches exports as `is_verified`."""
    file_id, det_id = _one_file(db)

    bulk_relabel_detections(
        BulkRelabelRequest(
            detection_ids=[det_id], label="false detection"
        ),
        db,
    )

    db.expire_all()
    assert db.get(File, file_id).verified is True
    # And the file is blank, because a rejected box cannot be its subject.
    assert db.get(File, file_id).observation_type == "blank"
