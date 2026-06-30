"""Tests for project duplication (crud.duplicate_project)."""

from app.api.crud.project import duplicate_project
from app.api.schemas.project import ProjectDuplicate
from app.models import Site
from app.models.deployment_queue import DeploymentQueue


def _params(name: str, **flags) -> ProjectDuplicate:
    base = dict(
        name=name,
        description="Duplicate",
        classification_model_id="SPECIESNET-v4-0-2-A",
        excluded_classes=["x"],
        copy_settings=True,
        copy_sites=True,
        copy_deployments=True,
    )
    base.update(flags)
    return ProjectDuplicate(**base)


def test_duplicate_copies_settings_sites_and_requeues_deployments(db):
    from tests.conftest import make_deployment, make_project, make_site

    source = make_project(
        db,
        name="Source",
        detection_threshold=0.7,
        independence_interval=600,
        taxonomic_rollup=False,
    )
    site = make_site(db, project_id=source.id, name="Cam 1")
    make_deployment(
        db,
        project_id=source.id,
        site_id=site.id,
        folder_path="/data/cam1",
        notes="hi",
    )

    new = duplicate_project(db, source.id, _params("Copy A"))
    assert new is not None
    # Settings carried over.
    assert new.detection_threshold == 0.7
    assert new.independence_interval == 600
    assert new.taxonomic_rollup is False
    # User-chosen fields from the request.
    assert new.name == "Copy A"
    assert new.classification_model_id == "SPECIESNET-v4-0-2-A"

    # Site copied with a new id.
    new_sites = db.query(Site).filter(Site.project_id == new.id).all()
    assert len(new_sites) == 1
    assert new_sites[0].id != site.id
    assert new_sites[0].name == "Cam 1"

    # Deployment re-queued, pointing at the new site, status pending.
    q = db.query(DeploymentQueue).filter(
        DeploymentQueue.project_id == new.id
    ).all()
    assert len(q) == 1
    assert q[0].folder_path == "/data/cam1"
    assert q[0].site_id == new_sites[0].id
    assert q[0].status == "pending"
    assert q[0].notes == "hi"


def test_duplicate_without_settings_uses_defaults(db):
    from tests.conftest import make_project

    source = make_project(db, name="Source2", detection_threshold=0.9)
    new = duplicate_project(
        db, source.id, _params("Copy B", copy_settings=False)
    )
    assert new is not None
    # Default, not the source's 0.9.
    assert new.detection_threshold == 0.5


def test_duplicate_without_sites_or_deployments(db):
    from tests.conftest import make_deployment, make_project, make_site

    source = make_project(db, name="Source3")
    site = make_site(db, project_id=source.id, name="Cam")
    make_deployment(
        db, project_id=source.id, site_id=site.id, folder_path="/d"
    )

    new = duplicate_project(
        db,
        source.id,
        _params("Copy C", copy_sites=False, copy_deployments=False),
    )
    assert new is not None
    assert db.query(Site).filter(Site.project_id == new.id).count() == 0
    assert (
        db.query(DeploymentQueue)
        .filter(DeploymentQueue.project_id == new.id)
        .count()
        == 0
    )


def test_duplicate_deployments_without_sites_are_site_less(db):
    from tests.conftest import make_deployment, make_project, make_site

    source = make_project(db, name="Source4")
    site = make_site(db, project_id=source.id, name="Cam")
    make_deployment(
        db, project_id=source.id, site_id=site.id, folder_path="/d"
    )

    new = duplicate_project(
        db, source.id, _params("Copy E", copy_sites=False, copy_deployments=True)
    )
    assert new is not None
    # No sites copied.
    assert db.query(Site).filter(Site.project_id == new.id).count() == 0
    # Deployment re-queued but with no site assignment.
    q = db.query(DeploymentQueue).filter(
        DeploymentQueue.project_id == new.id
    ).all()
    assert len(q) == 1
    assert q[0].site_id is None


def test_duplicate_missing_source_returns_none(db):
    assert duplicate_project(db, "nope", _params("Copy D")) is None
