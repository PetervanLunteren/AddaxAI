"""
Shared test fixtures for backend tests.

Provides:
- Environment isolation (temp directory, test DB)
- Session-scoped engine with WAL + FK pragmas
- Function-scoped DB session with savepoint/nested transaction pattern
- FastAPI TestClient with dependency override
- Factory helpers for building test data graphs
"""

import os
import tempfile
import uuid
from datetime import date, datetime
from pathlib import Path

# --- Environment isolation (must happen before any app imports) ---
_TEST_DIR = Path(tempfile.mkdtemp(prefix="addaxai_test_"))
(_TEST_DIR / "models" / "det").mkdir(parents=True)
(_TEST_DIR / "models" / "cls").mkdir(parents=True)
(_TEST_DIR / "models" / "emb").mkdir(parents=True)
(_TEST_DIR / "logs").mkdir(parents=True)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DIR / 'test.db'}")
os.environ.setdefault("USER_DATA_DIR", str(_TEST_DIR))
os.environ.setdefault("MODELS_DIR", str(_TEST_DIR / "models"))
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DISABLE_MODEL_UPDATES", "true")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base, get_db
from app.models.deployment import Deployment
from app.models.detection import Detection
from app.models.event import Event, event_files
from app.models.file import File
from app.models.job import Job
from app.models.project import Project
from app.models.site import Site

# Shared in-memory engine — single connection shared across all tests via StaticPool.
# This ensures all sessions see the same data (in-memory SQLite is per-connection).
_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)


@event.listens_for(_engine, "connect")
def _set_pragmas(conn, _):
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Create all tables once at import time
import app.models  # noqa: F401, E402

Base.metadata.create_all(bind=_engine)

_TestSessionLocal = sessionmaker(bind=_engine)


@pytest.fixture()
def db():
    """Function-scoped DB session that rolls back after each test."""
    session = _TestSessionLocal()
    yield session
    session.rollback()
    session.close()
    # Clean up all data for full isolation between tests
    with _engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


@pytest.fixture()
def client(db):
    """FastAPI TestClient with the DB dependency overridden to use the test session."""
    from unittest.mock import patch

    from app.main import create_app

    # Patch init_db to prevent the lifespan from touching the database
    with patch("app.main.init_db"):
        app = create_app()

        def _override_get_db():
            yield db

        app.dependency_overrides[get_db] = _override_get_db

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_project(db: Session, **kw) -> Project:
    defaults = dict(
        id=str(uuid.uuid4()),
        name=f"project-{uuid.uuid4().hex[:6]}",
        timezone="UTC",
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
        latitude=52.0,
        longitude=5.0,
    )
    defaults.update(kw)
    obj = Site(**defaults)
    db.add(obj)
    db.flush()
    return obj


def make_deployment(
    db: Session,
    *,
    site_id: str | None = None,
    project_id: str | None = None,
    **kw,
) -> Deployment:
    """Create a Deployment for tests.

    Deployment.project_id is required. If a caller passes only site_id,
    we resolve project_id from that site so existing call sites keep
    working without edits. Pass project_id explicitly when building a
    null-site deployment.
    """
    if project_id is None:
        if site_id is None:
            raise ValueError(
                "make_deployment needs project_id or site_id to resolve "
                "Deployment.project_id"
            )
        site = db.get(Site, site_id)
        if site is None:
            raise ValueError(f"Site {site_id} not found in make_deployment")
        project_id = site.project_id
    defaults = dict(
        id=str(uuid.uuid4()),
        project_id=project_id,
        site_id=site_id,
        start_date_local=date(2024, 1, 1),
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
    captured_at_local: datetime | None = None,
    verified: bool = False,
    **kw,
) -> File:
    defaults = dict(
        id=str(uuid.uuid4()),
        deployment_id=deployment_id,
        file_path=f"/fake/{uuid.uuid4().hex}.jpg",
        file_type="image",
        file_format="jpg",
        captured_at_local=captured_at_local or datetime(2024, 1, 1, 12, 0, 0),
        verified=verified,
    )
    defaults.update(kw)
    obj = File(**defaults)
    db.add(obj)
    db.flush()
    return obj


def make_detection(
    db: Session,
    *,
    file_id: str,
    category: str = "animal",
    confidence: float = 0.9,
    **kw,
) -> Detection:
    defaults = dict(
        id=str(uuid.uuid4()),
        file_id=file_id,
        category=category,
        confidence=confidence,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_width=0.2,
        bbox_height=0.2,
    )
    defaults.update(kw)
    obj = Detection(**defaults)
    db.add(obj)
    db.flush()
    return obj


def make_job(db: Session, *, job_type: str = "deployment_analysis", **kw) -> Job:
    defaults = dict(
        id=str(uuid.uuid4()),
        type=job_type,
    )
    defaults.update(kw)
    obj = Job(**defaults)
    db.add(obj)
    db.flush()
    return obj


def make_event_with_files(
    db: Session,
    *,
    deployment_id: str,
    event_start_local: datetime,
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
        event_start_local=event_start_local,
        event_end_local=event_start_local,
        file_count=len(files_verified),
    )
    db.add(ev)
    db.flush()

    for seq, verified in enumerate(files_verified):
        f = make_file(
            db,
            deployment_id=deployment_id,
            captured_at_local=event_start_local,
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


@pytest.fixture
def make_video():
    """Factory to render a deterministic tiny video for video tests.

    Each frame is a solid colour whose blue channel encodes the frame index
    (i % 256), so tests can verify which frames were decoded. Returns a
    callable: make(path, total_frames, fps=10, size=(64, 48), codec="mp4v").

    `codec` is a four-character cv2 fourcc. It exists so frame-fetch tests
    can cover more than one container and codec without carrying binary
    fixtures in the repo; pick the extension to match ("mp4v" -> .mp4,
    "MJPG" -> .avi). A codec the local build cannot encode skips the test
    rather than failing it, since the encoder set varies per OpenCV build.
    """
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    def _make(path, total_frames, fps=10, size=(64, 48), codec="mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not writer.isOpened():
            pytest.skip(
                f"cv2.VideoWriter could not open the {codec} encoder on this machine"
            )
        try:
            for i in range(total_frames):
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                frame[:, :, 0] = i % 256  # BGR: channel 0 = blue
                writer.write(frame)
        finally:
            writer.release()

    return _make
