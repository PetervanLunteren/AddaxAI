"""Tests for Deployment.site_id being optional and Deployment.project_id
being the authoritative project linkage."""

from sqlalchemy.exc import IntegrityError

from app.models.deployment import Deployment
from tests.conftest import make_deployment, make_project, make_site


def test_create_deployment_with_null_site(db):
    """A deployment can be created without a site (batch / unknown location)."""
    p = make_project(db)
    dep = make_deployment(db, project_id=p.id, site_id=None)
    db.expire_all()
    loaded = db.get(Deployment, dep.id)
    assert loaded is not None
    assert loaded.project_id == p.id
    assert loaded.site_id is None
    assert loaded.site is None


def test_project_id_required(db):
    """project_id is NOT NULL; creating a deployment without it raises."""
    import pytest

    p = make_project(db)
    s = make_site(db, project_id=p.id)
    # Bypass make_deployment's resolver: feed site_id only, blow out
    # project_id to simulate a buggy caller.
    obj = Deployment(
        site_id=s.id,
        start_date_local=make_deployment(db, site_id=s.id).start_date_local,
    )
    db.add(obj)
    with pytest.raises(IntegrityError):
        db.flush()


def test_site_delete_sets_null_not_cascade(db):
    """Deleting a site nulls out its deployments' site_id instead of deleting them."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d_with_site = make_deployment(db, site_id=s.id)
    d_without_site = make_deployment(db, project_id=p.id, site_id=None)

    db.delete(s)
    db.flush()
    db.expire_all()

    kept = db.get(Deployment, d_with_site.id)
    other = db.get(Deployment, d_without_site.id)
    assert kept is not None
    assert kept.site_id is None, "SET NULL on site delete"
    assert kept.project_id == p.id
    assert other is not None
    assert other.site_id is None


def test_project_delete_cascades_to_deployments(db):
    """Deleting a project still cascades to its deployments, including null-site ones."""
    p = make_project(db)
    s = make_site(db, project_id=p.id)
    d_sited = make_deployment(db, site_id=s.id)
    d_null = make_deployment(db, project_id=p.id, site_id=None)

    db.delete(p)
    db.flush()
    db.expire_all()

    assert db.get(Deployment, d_sited.id) is None
    assert db.get(Deployment, d_null.id) is None
