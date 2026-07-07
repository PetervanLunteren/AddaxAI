"""Tests for SQLAlchemy model constraints and defaults."""

from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.detection import Detection
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_job,
    make_project,
    make_site,
)


def test_project_defaults(db):
    p = make_project(db)
    assert p.detection_threshold == 0.2
    assert p.classification_gate == 0.1
    assert p.event_smoothing is True
    assert p.taxonomic_rollup is True
    assert p.independence_interval == 1800


def test_project_unique_name(db):
    make_project(db, name="unique-test")
    with pytest.raises(IntegrityError):
        make_project(db, name="unique-test")


def test_project_cascade_deletes_sites(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    site_id = s.id
    db.delete(p)
    db.flush()
    from app.models.site import Site
    assert db.get(Site, site_id) is None


def test_site_unique_name_per_project(db):
    p1 = make_project(db)
    p2 = make_project(db)
    make_site(db, project_id=p1.id, name="shared-name")
    # Same name in different project is OK
    make_site(db, project_id=p2.id, name="shared-name")
    # Same name in same project is NOT OK
    with pytest.raises(IntegrityError):
        make_site(db, project_id=p1.id, name="shared-name")


def test_site_delete_nulls_out_deployment_site_id(db):
    """Deleting a site should null out its deployments' site_id (SET NULL),
    not cascade-delete them. Deployments are owned by the project, not
    the site; nullable site_id is the supported state for
    deployment-agnostic batches."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    dep_id = d.id
    db.delete(s)
    db.flush()
    db.expire_all()
    from app.models.deployment import Deployment

    kept = db.get(Deployment, dep_id)
    assert kept is not None, "deployment should survive site deletion"
    assert kept.site_id is None
    assert kept.project_id == p.id


def test_deployment_cascade_deletes_files(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id)
    file_id = f.id
    db.delete(d)
    db.flush()
    from app.models.file import File
    assert db.get(File, file_id) is None


def test_file_cascade_deletes_detections(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id)
    det = make_detection(db, file_id=f.id)
    det_id = det.id
    db.delete(f)
    db.flush()
    assert db.get(Detection, det_id) is None


def test_detection_defaults(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    f = make_file(db, deployment_id=d.id)
    det = make_detection(db, file_id=f.id)
    assert det.verified is False


def test_foreign_key_enforcement(db):
    with pytest.raises(IntegrityError):
        make_detection(db, file_id="nonexistent-file-id")


def test_event_files_association(db):
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d = make_deployment(db, site_id=s.id)
    ev = make_event_with_files(
        db,
        deployment_id=d.id,
        event_start_local=datetime(2024, 1, 1, 12, 0),
        files_verified=[False, True],
    )
    assert ev.file_count == 2
    assert len(ev.files) == 2


def test_job_defaults(db):
    j = make_job(db)
    assert j.status == "pending"
    assert j.progress_current == 0
