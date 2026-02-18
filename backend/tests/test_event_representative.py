"""
Tests for event representative file selection and storage.

Verifies that:
- _select_representative_file picks the best file using shared scoring
- _create_event stores the representative_file_id
- generate_events_for_project loads detections and stores representatives
- get_events_by_project reads representative_file_id from DB (not computed)
"""

import uuid
from datetime import datetime
from unittest.mock import patch

from sqlalchemy import insert

from app.api.crud.event import (
    _create_event,
    _select_representative_file,
    generate_events_for_project,
    get_events_by_project,
)
from app.models.detection import Detection
from app.models.event import event_files
from app.models.file import File
from tests.conftest import (
    make_deployment,
    make_file,
    make_project,
    make_site,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_detection(db, file_id: str, category: str, confidence: float) -> Detection:
    det = Detection(
        id=str(uuid.uuid4()),
        file_id=file_id,
        category=category,
        confidence=confidence,
        bbox_x=0.0,
        bbox_y=0.0,
        bbox_width=0.1,
        bbox_height=0.1,
    )
    db.add(det)
    db.flush()
    return det


def _setup(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = make_deployment(db, site_id=site.id)
    return project, dep


# ---------------------------------------------------------------------------
# _select_representative_file
# ---------------------------------------------------------------------------


def test_select_representative_empty():
    assert _select_representative_file([]) is None


def test_select_representative_single_file(db):
    """Single file — always selected."""
    _, dep = _setup(db)
    f = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1))
    _add_detection(db, f.id, "animal", 0.9)

    # Reload to get detections relationship
    db.expire(f)
    f = db.query(File).filter(File.id == f.id).first()

    result = _select_representative_file([f])
    assert result == f.id


def test_select_representative_picks_highest_scoring(db):
    """File with highest animal confidence sum wins."""
    _, dep = _setup(db)

    f1 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1))
    f2 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1, 0, 1))

    # f1 has low animal confidence
    _add_detection(db, f1.id, "animal", 0.4)

    # f2 has high animal confidence (two detections)
    _add_detection(db, f2.id, "animal", 0.9)
    _add_detection(db, f2.id, "animal", 0.8)

    # Reload
    from sqlalchemy.orm import joinedload
    files = (
        db.query(File)
        .options(joinedload(File.detections))
        .filter(File.id.in_([f1.id, f2.id]))
        .all()
    )

    result = _select_representative_file(files)
    assert result == f2.id


def test_select_representative_ignores_non_animal(db):
    """Non-animal detections are ignored for scoring."""
    _, dep = _setup(db)

    f1 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1))
    f2 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1, 0, 1))

    # f1 has high confidence but it's "person", not "animal"
    _add_detection(db, f1.id, "person", 0.95)

    # f2 has moderate animal confidence
    _add_detection(db, f2.id, "animal", 0.5)

    from sqlalchemy.orm import joinedload
    files = (
        db.query(File)
        .options(joinedload(File.detections))
        .filter(File.id.in_([f1.id, f2.id]))
        .all()
    )

    result = _select_representative_file(files)
    assert result == f2.id


def test_select_representative_below_threshold_fallback(db):
    """All detections below threshold — falls back to first file."""
    _, dep = _setup(db)

    f1 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1))
    f2 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1, 0, 1))

    _add_detection(db, f1.id, "animal", 0.1)  # below 0.3 threshold
    _add_detection(db, f2.id, "animal", 0.2)  # below 0.3 threshold

    from sqlalchemy.orm import joinedload
    files = (
        db.query(File)
        .options(joinedload(File.detections))
        .filter(File.id.in_([f1.id, f2.id]))
        .all()
    )

    # With no qualifying scores, get_sharpest is called with fallback_keys.
    # Since images don't exist on disk, get_sharpest falls back to first key.
    # The ultimate fallback in _select_representative_file is files[0].id.
    result = _select_representative_file(files)
    assert result in [f1.id, f2.id]


# ---------------------------------------------------------------------------
# _create_event stores representative
# ---------------------------------------------------------------------------


def test_create_event_stores_representative(db):
    """_create_event should populate representative_file_id."""
    _, dep = _setup(db)

    f = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1))
    _add_detection(db, f.id, "animal", 0.9)

    from sqlalchemy.orm import joinedload
    f = db.query(File).options(joinedload(File.detections)).filter(File.id == f.id).first()

    event = _create_event(db, dep.id, [f])
    assert event.representative_file_id == f.id


# ---------------------------------------------------------------------------
# generate_events_for_project
# ---------------------------------------------------------------------------


def test_generate_events_stores_representative(db):
    """Full pipeline: generate_events_for_project stores representative_file_id."""
    project, dep = _setup(db)

    f1 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1, 12, 0))
    f2 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1, 12, 1))

    _add_detection(db, f1.id, "animal", 0.5)
    _add_detection(db, f2.id, "animal", 0.9)

    db.commit()

    count = generate_events_for_project(db, project.id)
    assert count == 1

    from app.models import Event
    events = db.query(Event).filter(Event.deployment_id == dep.id).all()
    assert len(events) == 1
    assert events[0].representative_file_id == f2.id


# ---------------------------------------------------------------------------
# get_events_by_project
# ---------------------------------------------------------------------------


def test_get_events_returns_stored_representative(db):
    """get_events_by_project should return representative_file_id from the DB."""
    project, dep = _setup(db)

    f1 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1, 12, 0))
    f2 = make_file(db, deployment_id=dep.id, timestamp=datetime(2024, 6, 1, 12, 1))

    _add_detection(db, f1.id, "animal", 0.5)
    _add_detection(db, f2.id, "animal", 0.9)

    db.commit()

    generate_events_for_project(db, project.id)

    summaries = get_events_by_project(db, project.id)
    assert len(summaries) == 1
    assert summaries[0]["representative_file_id"] == f2.id
