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


def _embed(session, detection_id: str, vector=None) -> None:
    """Attach an embedding. ``vector`` (any length) is stored with its
    own L2 norm so the loader renormalises it to a unit vector; default
    is a zero vector (fine for tests that only check presence)."""
    v = (
        np.zeros(4, dtype=np.float32)
        if vector is None
        else np.asarray(vector, dtype=np.float32)
    )
    norm = float(np.linalg.norm(v)) or 1.0
    session.add(
        DetectionEmbedding(
            id=str(uuid.uuid4()),
            detection_id=detection_id,
            embedding_model_id="DINOV2-VITB14",
            vector=v.astype(np.float16).tobytes(),
            dimension=len(v),
            l2_norm=norm,
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
    vector=None,
):
    """Create one event (if new), a file in it, and a detection on that
    file. Optionally embed the detection (with an optional vector)."""
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
        _embed(session, d.id, vector)
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


# ── Event ordering by similarity ─────────────────────────────────────
# The FAISS-dependent greedy walk is isolated in `_greedy_order`, so the
# grouping / representative / assembly logic is tested here by patching
# it (no FAISS needed). The real walk is exercised in the faiss-gated
# tests further down.

from unittest.mock import MagicMock  # noqa: E402

from app.ml.inference.similarity_script import (  # noqa: E402
    _order_events_by_similarity,
)


def _meta(event_id, seq, start, confidence):
    return {
        "event_id": event_id,
        "event_sequence": seq,
        "event_start_local": start,
        "confidence": confidence,
    }


def test_order_events_by_similarity_assembly(monkeypatch):
    # Events by time desc: C (03) , B (02), A (01). A and B are embedded,
    # C is not. A has two detections; its representative must be the
    # most-confident one (a1, conf 0.9), not a0 (0.5).
    metas = [
        _meta("A", 1, "2024-01-01T00:00", 0.9),  # 0: a1 (embedded, high conf)
        _meta("A", 0, "2024-01-01T00:00", 0.5),  # 1: a0 (embedded, low conf)
        _meta("B", 0, "2024-01-02T00:00", 0.8),  # 2: b0 (embedded)
        _meta("C", 0, "2024-01-03T00:00", 0.7),  # 3: c0 (NOT embedded)
        _meta(None, 0, None, 0.6),               # 4: no event
    ]
    det_ids = ["a1", "a0", "b0", "c0", "n0"]
    va, vb = [1.0, 0.0], [0.0, 1.0]
    vector_by_id = {
        "a1": np.array(va, dtype=np.float32),
        "a0": np.array([0.0, 0.0], dtype=np.float32),
        "b0": np.array(vb, dtype=np.float32),
    }

    captured = {}

    def fake_greedy_order(mat):
        captured["mat"] = mat
        # rep_eids are the embedded events in time order: [B, A].
        # Return them reversed so we can see the walk order is applied.
        return [1, 0]

    monkeypatch.setattr(
        "app.ml.inference.similarity_script._greedy_order", fake_greedy_order
    )

    order = _order_events_by_similarity(det_ids, metas, vector_by_id)

    # Walk said [B, A] reversed -> events emitted A then B; within A the
    # sequence order (a0 seq0 before a1 seq1); then the rep-less event C;
    # then the no-event detection last.
    assert order == [1, 0, 2, 3, 4]
    # Representative matrix was built from [B, A]'s most-confident crops.
    mat = captured["mat"]
    assert np.allclose(mat[0], vb)  # B's representative
    assert np.allclose(mat[1], va)  # A's representative = a1, not a0


def test_order_events_by_similarity_no_reps_is_time_order(monkeypatch):
    # No event has an embedded detection -> no walk, pure time order.
    metas = [
        _meta("A", 0, "2024-01-01T00:00", 0.9),
        _meta("B", 0, "2024-01-02T00:00", 0.8),
    ]
    det_ids = ["a", "b"]
    called = MagicMock()
    monkeypatch.setattr(
        "app.ml.inference.similarity_script._greedy_order", called
    )
    order = _order_events_by_similarity(det_ids, metas, {})
    called.assert_not_called()
    assert order == [1, 0]  # newest event (B) first


# ── FAISS-gated: real greedy walk ────────────────────────────────────


def test_greedy_order_keeps_identical_vectors_adjacent():
    faiss = pytest.importorskip("faiss")
    from app.ml.inference.similarity_script import _greedy_order

    del faiss  # only needed for the skip guard
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    order = _greedy_order(vectors)
    # The two identical vectors (indices 0 and 2) must be neighbours in
    # the walk; the distinct one (1) cannot sit between them.
    pos = {idx: p for p, idx in enumerate(order)}
    assert abs(pos[0] - pos[2]) == 1


def test_event_sort_similarity_groups_similar_events(sort_db):
    pytest.importorskip("faiss")
    db_path, s = sort_db
    p = make_project(s, embedding_model_id="DINOV2-VITB14")
    dep = make_deployment(s, project_id=p.id)
    # Events A and B look alike; C is different; D has no embedding.
    _detection_in_event(
        s, dep.id, event_id="A", start=datetime(2024, 1, 1, 12),
        embed=True, vector=[1.0, 0.0, 0.0, 0.0],
    )
    _detection_in_event(
        s, dep.id, event_id="B", start=datetime(2024, 1, 2, 12),
        embed=True, vector=[1.0, 0.0, 0.0, 0.0],
    )
    _detection_in_event(
        s, dep.id, event_id="C", start=datetime(2024, 1, 3, 12),
        embed=True, vector=[0.0, 1.0, 0.0, 0.0],
    )
    d_plain = _detection_in_event(
        s, dep.id, event_id="D", start=datetime(2024, 1, 4, 12), embed=False,
    )
    s.commit()

    result = do_sort(db_path, p.id, {"sort": "events", "filters": {}})
    event_seq = [d["event_id"] for d in result["detections"]]

    # Look-alike events A and B are adjacent (C not wedged between them).
    ia, ib = event_seq.index("A"), event_seq.index("B")
    assert abs(ia - ib) == 1
    # The non-embedded event D is still present (rep-less, at the tail).
    assert d_plain.detection_id in {
        d["detection_id"] for d in result["detections"]
    }
    assert event_seq[-1] == "D"
