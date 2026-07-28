"""crud.site.create_site auto-derives the project's camera timezone.

A new project starts with no timezone; the first sited site sets it from
its coordinates. Later sites never change it, and an already-set timezone
is never overwritten (KISS: first site wins).
"""

from app.api.crud import site as site_crud
from app.api.schemas.site import SiteCreate
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
