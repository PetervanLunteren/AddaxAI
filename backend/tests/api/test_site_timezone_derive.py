"""crud.site.create_site auto-derives the project's camera timezone.

A new project starts with no timezone; the first sited site sets it from
its coordinates. Later sites never change it, and an already-set timezone
is never overwritten (KISS: first site wins).
"""

import logging

from app.api.crud import site as site_crud
from app.api.schemas.site import SiteCreate
from app.models import Site
from tests.conftest import make_project

# Serengeti / Yellowstone — stable labels under TimezoneFinderL.
_SERENGETI = (-2.33, 34.83)
_YELLOWSTONE = (44.6, -110.5)


def _create_site(db, project_id, lat, lon, name="site"):
    return site_crud.create_site(
        db,
        SiteCreate(project_id=project_id, name=name, latitude=lat, longitude=lon),
    )


def test_first_site_sets_unset_project_timezone(db):
    project = make_project(db, timezone=None)
    _create_site(db, project.id, *_SERENGETI)
    db.refresh(project)
    assert project.timezone == "Africa/Dar_es_Salaam"


def test_second_site_does_not_change_timezone(db):
    project = make_project(db, timezone=None)
    _create_site(db, project.id, *_SERENGETI, name="a")
    db.refresh(project)
    first = project.timezone
    _create_site(db, project.id, *_YELLOWSTONE, name="b")
    db.refresh(project)
    assert project.timezone == first  # unchanged by the second site


def test_existing_timezone_is_not_overwritten(db):
    project = make_project(db, timezone="UTC")
    _create_site(db, project.id, *_SERENGETI)
    db.refresh(project)
    assert project.timezone == "UTC"


def _broken_lookup(lat, lon):
    raise RuntimeError("timezone data missing")


def test_a_failed_lookup_leaves_the_timezone_unset_and_keeps_the_site(
    db, monkeypatch, caplog
):
    """The lookup is a convenience on top of a site that is already saved.
    A broken lookup (seen as a 500 on a packaged Windows build) must not
    turn a saved site into a reported failure; it is logged instead."""
    monkeypatch.setattr(site_crud, "tz_from_coords", _broken_lookup)
    project = make_project(db, timezone=None)

    with caplog.at_level(logging.ERROR):
        site = _create_site(db, project.id, *_SERENGETI)

    assert db.get(Site, site.id) is not None
    db.refresh(project)
    assert project.timezone is None
    failures = [r for r in caplog.records if "left unset" in r.getMessage()]
    assert len(failures) == 1
    assert failures[0].exc_info is not None


def test_post_site_answers_201_when_the_lookup_fails(client, db, monkeypatch):
    monkeypatch.setattr(site_crud, "tz_from_coords", _broken_lookup)
    project = make_project(db, timezone=None)

    resp = client.post("/api/sites", json={
        "name": "MS site",
        "project_id": project.id,
        "latitude": -35.69718952312521,
        "longitude": 148.98372595411246,
        "elevation_m": 600,
        "habitat_type": "Dry Sclerophyll",
    })

    assert resp.status_code == 201
