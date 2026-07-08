"""Event sort works without embeddings.

The labels grid's "Sort by event" orders detections by event / capture
time, which needs no embeddings. Historically the shared sort subprocess
loaded everything ``FROM detection_embeddings``, so event sort silently
dropped detections that were never embedded (a project with the
embedding model off, or the below-gate tail). ``do_sort`` now takes a
metadata-only load path for the non-embedding sort modes.

These tests drive the subprocess helpers in-process against a temp file
SQLite DB (the helpers open it read-only, like the real subprocess).
FAISS is never needed here: the metadata path skips it, and the
similarity regression is checked at the loader level, not through
``do_sort``.
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.ml.inference.similarity_script import (
    _load_embeddings,
    _load_metadata,
    do_sort,
)
from app.models.detection_embedding import DetectionEmbedding
from app.models.event import Event, event_files
from tests.conftest import make_deployment, make_detection, make_file, make_project

# do_sort imports its sibling `observation_sort` by bare name (it runs as
# a subprocess in production, where its own dir is on sys.path[0]). Make
# that sibling importable when we call do_sort in-process here.
_INFERENCE_DIR = Path(__file__).resolve().parents[2] / "app" / "ml" / "inference"
sys.path.insert(0, str(_INFERENCE_DIR))


@pytest.fixture
def sort_db(tmp_path):
    """A file-based SQLite DB with the full schema and a bound session.

    File-based (not the shared in-memory test engine) because the sort
    helpers open the DB by path in read-only mode. Plain rollback journal
    (no WAL) so a committed write is visible to that read-only handle.
    """
    path = tmp_path / "sort.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield str(path), session
    finally:
        session.close()
        engine.dispose()


def _embed(session, detection_id: str) -> None:
    session.add(
        DetectionEmbedding(
            id=str(uuid.uuid4()),
            detection_id=detection_id,
            embedding_model_id="DINOV2-VITB14",
            vector=np.zeros(4, dtype=np.float16).tobytes(),
            dimension=4,
            l2_norm=1.0,
        )
    )


def _detection_in_event(
    session,
    deployment_id: str,
    *,
    event_id: str,
    start: datetime,
    seq: int = 0,
    confidence: float = 0.9,
    verified: bool = False,
    embed: bool = False,
):
    """Create one event (if new), a file in it, and a detection on that
    file. Optionally embed the detection."""
    ev = session.get(Event, event_id)
    if ev is None:
        session.add(
            Event(
                id=event_id,
                deployment_id=deployment_id,
                event_start_local=start,
                event_end_local=start,
                file_count=0,
            )
        )
        session.flush()
    f = make_file(
        session,
        deployment_id=deployment_id,
        captured_at_local=start,
        verified=verified,
        width_px=1920,
        height_px=1080,
    )
    session.execute(
        event_files.insert().values(
            event_id=event_id, file_id=f.id, sequence_number=seq
        )
    )
    d = make_detection(
        session, file_id=f.id, confidence=confidence, verified=verified
    )
    if embed:
        _embed(session, d.id)
    session.flush()
    return d


def test_metadata_load_includes_non_embedded_detections(sort_db):
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    embedded = _detection_in_event(
        s, dep.id, event_id="ev-b", start=datetime(2024, 1, 2, 12), embed=True
    )
    plain = _detection_in_event(
        s, dep.id, event_id="ev-a", start=datetime(2024, 1, 1, 12), embed=False
    )
    s.commit()

    # Metadata path sees both; embedding path sees only the embedded one.
    ids_meta, _ = _load_metadata(db_path, p.id, {})
    assert set(ids_meta) == {embedded.id, plain.id}

    _, ids_emb, _ = _load_embeddings(db_path, p.id, {})
    assert set(ids_emb) == {embedded.id}


def test_event_sort_returns_non_embedded_in_event_order(sort_db):
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    older = _detection_in_event(
        s, dep.id, event_id="ev-old", start=datetime(2024, 1, 1, 12), embed=False
    )
    newer = _detection_in_event(
        s, dep.id, event_id="ev-new", start=datetime(2024, 1, 2, 12), embed=True
    )
    s.commit()

    result = do_sort(db_path, p.id, {"sort": "events", "filters": {}})
    ids = [d["detection_id"] for d in result["detections"]]

    # Newest event first; the non-embedded detection is present.
    assert ids == [newer.id, older.id]
    assert result["total_detections"] == 2
    # Metadata path carries no neighbour signal.
    plain = next(d for d in result["detections"] if d["detection_id"] == older.id)
    assert plain["neighbor_agreement"] is None


def test_metadata_load_applies_project_floor(sort_db):
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    low_unverified = _detection_in_event(
        s, dep.id, event_id="e1", start=datetime(2024, 1, 1, 12),
        confidence=0.05, verified=False,
    )
    low_verified = _detection_in_event(
        s, dep.id, event_id="e2", start=datetime(2024, 1, 2, 12),
        confidence=0.05, verified=True,
    )
    s.commit()

    ids, _ = _load_metadata(db_path, p.id, {"project_floor": 0.2})
    # Below-floor unverified is excluded; verified overrides the floor.
    assert low_verified.id in ids
    assert low_unverified.id not in ids
