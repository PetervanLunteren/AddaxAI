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
    ids_meta, _, _ = _load_metadata(db_path, p.id, {})
    assert set(ids_meta) == {embedded.id, plain.id}

    _, ids_emb, _, _ = _load_embeddings(db_path, p.id, {})
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

    ids, _, _ = _load_metadata(db_path, p.id, {"project_floor": 0.2})
    # Below-floor unverified is excluded; verified overrides the floor.
    assert low_verified.id in ids
    assert low_unverified.id not in ids


def test_sort_caps_to_newest_and_reports_uncapped_total(sort_db):
    """Over the cap, do_sort loads the newest `cap` by capture time and
    reports the uncapped total (no error)."""
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    _detection_in_event(s, dep.id, event_id="e1", start=datetime(2024, 1, 1, 12))
    d2 = _detection_in_event(s, dep.id, event_id="e2", start=datetime(2024, 1, 2, 12))
    d3 = _detection_in_event(s, dep.id, event_id="e3", start=datetime(2024, 1, 3, 12))
    s.commit()

    result = do_sort(
        db_path, p.id, {"sort": "events", "filters": {}, "max_detections": 2}
    )
    ids = [d["detection_id"] for d in result["detections"]]
    assert ids == [d3.id, d2.id]  # newest two, newest event first
    assert result["total_detections"] == 2
    assert result["total_matching"] == 3


def test_sort_under_cap_matches_total(sort_db):
    """Under the cap, everything loads and total_matching == total_detections."""
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    _detection_in_event(s, dep.id, event_id="e1", start=datetime(2024, 1, 1, 12))
    _detection_in_event(s, dep.id, event_id="e2", start=datetime(2024, 1, 2, 12))
    s.commit()

    result = do_sort(db_path, p.id, {"sort": "events", "filters": {}})
    assert result["total_detections"] == 2
    assert result["total_matching"] == 2


def test_sort_verified_filter_scopes_the_cap(sort_db):
    """The cap and total_matching count only the current verified-filter
    pool: with verified=False, verified detections are excluded entirely."""
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    u1 = _detection_in_event(s, dep.id, event_id="e1", start=datetime(2024, 1, 1, 12))
    u2 = _detection_in_event(s, dep.id, event_id="e2", start=datetime(2024, 1, 2, 12))
    _detection_in_event(
        s, dep.id, event_id="e3", start=datetime(2024, 1, 3, 12), verified=True
    )
    s.commit()

    result = do_sort(
        db_path, p.id, {"sort": "events", "filters": {"verified": False}}
    )
    ids = {d["detection_id"] for d in result["detections"]}
    assert ids == {u1.id, u2.id}
    assert result["total_matching"] == 2


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

    has_embedding = set(vector_by_id)
    requested: list[str] = []

    def load_vectors(ids):
        requested.extend(ids)
        return {i: vector_by_id[i] for i in ids}

    order = _order_events_by_similarity(
        det_ids, metas, has_embedding, load_vectors
    )

    # Walk said [B, A] reversed -> events emitted A then B; within A the
    # sequence order (a0 seq0 before a1 seq1); then the rep-less event C;
    # then the no-event detection last.
    assert order == [1, 0, 2, 3, 4]
    # Representative matrix was built from [B, A]'s most-confident crops.
    mat = captured["mat"]
    assert np.allclose(mat[0], vb)  # B's representative
    assert np.allclose(mat[1], va)  # A's representative = a1, not a0
    # Only the two representatives were loaded, not a0 (embedded but not
    # its event's rep) or the rep-less / no-event detections.
    assert set(requested) == {"a1", "b0"}


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
    order = _order_events_by_similarity(det_ids, metas, set(), lambda ids: {})
    called.assert_not_called()
    assert order == [1, 0]  # newest event (B) first


def test_order_events_by_similarity_caps_walk_at_max_embeddings(monkeypatch):
    # Three embedded events; with a FAISS budget of 2, only the newest two
    # are similarity-walked (and their vectors loaded); the third keeps its
    # baseline (chronological) place in the tail and isn't loaded.
    metas = [
        _meta("A", 0, "2024-01-01T00:00", 0.9),
        _meta("B", 0, "2024-01-02T00:00", 0.9),
        _meta("C", 0, "2024-01-03T00:00", 0.9),
    ]
    det_ids = ["a", "b", "c"]
    vecs = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
        "c": np.array([1.0, 1.0], dtype=np.float32),
    }
    requested: list[str] = []

    def load_vectors(ids):
        requested.extend(ids)
        return {i: vecs[i] for i in ids}

    # Identity walk over the (capped) rep set.
    monkeypatch.setattr(
        "app.ml.inference.similarity_script._greedy_order",
        lambda mat: list(range(len(mat))),
    )
    order = _order_events_by_similarity(
        det_ids, metas, set(vecs), load_vectors, max_embeddings=2
    )
    # Baseline is time-desc [C, B, A]; walk the first two, A falls to tail.
    assert order == [2, 1, 0]
    # Only the two walked reps were loaded — A's vector was over budget.
    assert set(requested) == {"c", "b"}


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


# ── No-embedding fallback: deployment grouping ───────────────────────

from observation_sort import order_events_by_deployment  # noqa: E402


def test_order_events_by_deployment_groups_cameras():
    # DEP1 has the newest event (01-05) so it leads; its two events stay
    # together (01-05 then 01-01) even though DEP2's event (01-03) is
    # chronologically between them.
    metas = [
        _meta_dep("DEP1", "E1a", "2024-01-05", 0),  # 0
        _meta_dep("DEP1", "E1b", "2024-01-01", 0),  # 1
        _meta_dep("DEP2", "E2a", "2024-01-03", 0),  # 2
    ]
    # Plain chronological would interleave the cameras ([0, 2, 1]);
    # deployment grouping keeps DEP1's two events together.
    assert order_events_by_deployment(metas) == [0, 1, 2]


def test_order_events_by_deployment_single_deployment_is_chronological():
    metas = [
        _meta_dep("D", "E1", "2024-01-01", 1),
        _meta_dep("D", "E1", "2024-01-01", 0),
        _meta_dep("D", "E2", "2024-01-02", 0),
    ]
    # One camera: plain chronological (event newest-first, within event
    # by ascending sequence) — E2 (01-02), then E1 (01-01) seq 0 then 1.
    assert order_events_by_deployment(metas) == [2, 1, 0]


def _meta_dep(deployment_id, event_id, start, seq):
    return {
        "deployment_id": deployment_id,
        "event_id": event_id,
        "event_start_local": start,
        "event_sequence": seq,
    }


def test_event_sort_no_embeddings_groups_by_deployment(sort_db):
    db_path, s = sort_db
    p = make_project(s)  # no embedding model
    dep1 = make_deployment(s, project_id=p.id)
    dep2 = make_deployment(s, project_id=p.id)
    # dep1's newest event is the most recent overall, so dep1 leads and
    # its two events stay together despite dep2's event falling between
    # them in time.
    d1_new = _detection_in_event(
        s, dep1.id, event_id="d1-new", start=datetime(2024, 1, 5, 12)
    )
    d1_old = _detection_in_event(
        s, dep1.id, event_id="d1-old", start=datetime(2024, 1, 1, 12)
    )
    d2 = _detection_in_event(
        s, dep2.id, event_id="d2", start=datetime(2024, 1, 3, 12)
    )
    s.commit()

    result = do_sort(db_path, p.id, {"sort": "events", "filters": {}})
    ids = [d["detection_id"] for d in result["detections"]]
    assert ids == [d1_new.id, d1_old.id, d2.id]


# ── Video detections are gated to the best frame ─────────────────────
# Only the best frame of a video is written to disk as a JPEG, so a
# detection on any other sampled frame has no image to crop. The grid used
# to list all of them and `crop_service` answered every one with the best
# frame cropped at a bbox from a different moment: a picture of wherever
# the animal used to be. See tests/test_crop_service.py for the other half.


def _video_with_detections(session, deployment_id, *, best_frame, frames):
    """One video file plus a detection on each of `frames`. Returns
    {frame_number: detection}."""
    f = make_file(
        session,
        deployment_id=deployment_id,
        file_type="video",
        file_format="mp4",
        file_path=f"/fake/{uuid.uuid4().hex}.mp4",
        best_frame_number=best_frame,
        best_frame_path=f"/fake/frame{best_frame:06d}.jpg",
    )
    out = {}
    for fn in frames:
        out[fn] = make_detection(session, file_id=f.id, frame_number=fn)
    session.flush()
    return out


def test_metadata_load_drops_video_detections_off_the_best_frame(sort_db):
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    # A raccoon walking across the scene: MegaDetector fires on every
    # sampled frame, but only frame 24 exists as a JPEG.
    dets = _video_with_detections(
        s, dep.id, best_frame=24, frames=[0, 24, 48, 72, 144]
    )
    s.commit()

    ids, _, _ = _load_metadata(db_path, p.id, {})

    assert set(ids) == {dets[24].id}


def test_metadata_load_keeps_every_image_detection(sort_db):
    """Images are never gated: `frame_number` is NULL on an image row and
    the file itself is the croppable surface."""
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    f = make_file(s, deployment_id=dep.id)
    a = make_detection(s, file_id=f.id)
    b = make_detection(s, file_id=f.id)
    s.commit()

    ids, _, _ = _load_metadata(db_path, p.id, {})

    assert set(ids) == {a.id, b.id}


def test_metadata_load_keeps_verified_detections_off_the_best_frame(sort_db):
    """A human decision must never end up out of reach. `rebuild_event_
    observations` lets a species verified on any frame into the counts, so
    the grid has to be able to show the card that count came from, even
    though its thumbnail will be missing."""
    db_path, s = sort_db
    p = make_project(s)
    dep = make_deployment(s, project_id=p.id)
    dets = _video_with_detections(s, dep.id, best_frame=24, frames=[24, 144])
    dets[144].verified = True
    s.commit()

    ids, _, _ = _load_metadata(db_path, p.id, {})

    assert set(ids) == {dets[24].id, dets[144].id}


def test_order_events_by_similarity_keeps_partial_event_intact(monkeypatch):
    # Event X has one embedded (x_emb, the representative) and one NON
    # embedded detection (x_plain); event Y is embedded. Both of X's
    # detections must stay together in the block, in sequence order,
    # even though x_plain has no vector.
    metas = [
        _meta("X", 0, "2024-01-02T00:00", 0.9),  # 0: x_emb (embedded)
        _meta("X", 1, "2024-01-02T00:00", 0.4),  # 1: x_plain (NOT embedded)
        _meta("Y", 0, "2024-01-01T00:00", 0.8),  # 2: y_emb (embedded)
    ]
    det_ids = ["x_emb", "x_plain", "y_emb"]
    vector_by_id = {
        "x_emb": np.array([1.0, 0.0], dtype=np.float32),
        "y_emb": np.array([0.0, 1.0], dtype=np.float32),
    }
    # Identity walk over rep events [X, Y].
    monkeypatch.setattr(
        "app.ml.inference.similarity_script._greedy_order",
        lambda mat: list(range(len(mat))),
    )
    order = _order_events_by_similarity(
        det_ids,
        metas,
        set(vector_by_id),
        lambda ids: {i: vector_by_id[i] for i in ids},
    )

    # X block first (both its detections, sequence order), then Y.
    assert order == [0, 1, 2]
    # The non-embedded detection is present and adjacent to its event's
    # embedded one (not split off to a tail).
    pos = {det_ids[i]: p for p, i in enumerate(order)}
    assert abs(pos["x_emb"] - pos["x_plain"]) == 1
