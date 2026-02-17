"""
Shared test fixtures for backend tests.

Provides an in-memory SQLite database with all tables created,
a session-per-test that rolls back after each test, and factory
helpers for building the Project → Site → Deployment → Event → File graph.
"""

import uuid
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.models.project import Project
from app.models.site import Site
from app.models.deployment import Deployment
from app.models.event import Event, event_files
from app.models.file import File


@pytest.fixture()
def db_engine():
    engine = create_engine("sqlite://", echo=False)

    @event.listens_for(engine, "connect")
    def _set_pragmas(conn, _):
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db(db_engine):
    session = sessionmaker(bind=db_engine)()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_project(db: Session, **kw) -> Project:
    defaults = dict(
        id=str(uuid.uuid4()),
        name=f"project-{uuid.uuid4().hex[:6]}",
    )
    defaults.update(kw)
    obj = Project(**defaults)
    db.add(obj)
    db.flush()
    return obj


def make_site(db: Session, *, project_id: str, **kw) -> Site:
    defaults = dict(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name=f"site-{uuid.uuid4().hex[:6]}",
    )
    defaults.update(kw)
    obj = Site(**defaults)
    db.add(obj)
    db.flush()
    return obj


def make_deployment(db: Session, *, site_id: str, **kw) -> Deployment:
    defaults = dict(
        id=str(uuid.uuid4()),
        site_id=site_id,
        start_date=date(2024, 1, 1),
    )
    defaults.update(kw)
    obj = Deployment(**defaults)
    db.add(obj)
    db.flush()
    return obj


def make_file(
    db: Session,
    *,
    deployment_id: str,
    timestamp: datetime,
    verified: bool = False,
    **kw,
) -> File:
    defaults = dict(
        id=str(uuid.uuid4()),
        deployment_id=deployment_id,
        file_path=f"/fake/{uuid.uuid4().hex}.jpg",
        file_type="image",
        timestamp=timestamp,
        verified=verified,
    )
    defaults.update(kw)
    obj = File(**defaults)
    db.add(obj)
    db.flush()
    return obj


def make_event_with_files(
    db: Session,
    *,
    deployment_id: str,
    start_time: datetime,
    files_verified: list[bool] | None = None,
    event_id: str | None = None,
) -> Event:
    """
    Create an event with associated files.

    files_verified: list of verified flags, one per file to create.
                    Defaults to [False] (one unverified file).
    """
    if files_verified is None:
        files_verified = [False]

    eid = event_id or str(uuid.uuid4())
    ev = Event(
        id=eid,
        deployment_id=deployment_id,
        start_time=start_time,
        end_time=start_time,
        file_count=len(files_verified),
    )
    db.add(ev)
    db.flush()

    for seq, verified in enumerate(files_verified):
        f = make_file(
            db,
            deployment_id=deployment_id,
            timestamp=start_time,
            verified=verified,
        )
        db.execute(
            event_files.insert().values(
                event_id=eid,
                file_id=f.id,
                sequence_number=seq,
            )
        )

    db.flush()
    return ev
