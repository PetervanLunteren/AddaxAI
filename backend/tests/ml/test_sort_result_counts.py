"""The three counts ``do_sort`` returns, and which one means "capped".

``total_matching`` is the uncapped pool, ``total_loaded`` is what this
run read (``min(total_matching, cap)``), and ``total_detections`` is what
the sort returned. Only the first two say whether the memory guard bit.

The UI read truncation off ``total_matching > total_detections``, which
holds for the similarity and event sorts (both return every row they
load) and is wrong for `suggestions`, which narrows to cohort members by
design. A project with 814 embedded detections and one 8-member cohort
showed "showing the newest 8 of 814, capped to stay responsive" on a pool
nowhere near the 20,000 cap. These tests pin all three counts per sort
mode so that comparison cannot be reintroduced.

FAISS is not installed in the backend venv (it lives in env-addaxai-base,
where the script really runs), so the suggestions path gets a numpy
stand-in for ``IndexFlatIP``. That index is exact brute-force inner
product, which numpy reproduces exactly at these sizes, so the stub
changes nothing about what is under test.
"""

import sys
import types
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.ml.inference.similarity_script import do_sort
from app.models.detection_embedding import DetectionEmbedding
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import make_deployment, make_detection, make_file, make_project

# do_sort imports its sibling `observation_sort` by bare name (it runs as
# a subprocess in production, where its own dir is on sys.path[0]). Make
# that sibling importable when we call do_sort in-process here.
_INFERENCE_DIR = Path(__file__).resolve().parents[2] / "app" / "ml" / "inference"
sys.path.insert(0, str(_INFERENCE_DIR))

CLS_MODEL = "TEST-CLS-v1"


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


class _NumpyFlatIP:
    """Exact inner-product index, the same contract as faiss.IndexFlatIP."""

    def __init__(self, dim: int):
        self._dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, vectors) -> None:
        self._vectors = np.vstack([self._vectors, np.asarray(vectors)])

    def search(self, queries, k):
        queries = np.atleast_2d(np.asarray(queries))
        sims = queries @ self._vectors.T
        # Descending by similarity, ties broken by index, exactly how a
        # flat index enumerates equal-scoring vectors.
        idxs = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(sims, idxs, axis=1), idxs


@pytest.fixture
def fake_faiss(monkeypatch):
    """Put a numpy-backed `faiss` in sys.modules for do_sort's local import."""
    module = types.ModuleType("faiss")
    module.IndexFlatIP = _NumpyFlatIP
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


def _embedded_detection(
    session, deployment_id: str, *, label: str, captured_at: datetime
):
    """One file + detection carrying `label`, with a unit embedding.

    Every detection gets the same vector, so each one's nearest
    neighbours are simply everything else. That makes the neighbour
    majority a function of the label mix alone, which is what the
    suggestions cohorting is about.
    """
    f = make_file(
        session,
        deployment_id=deployment_id,
        captured_at_local=captured_at,
        width_px=1920,
        height_px=1080,
    )
    d = make_detection(session, file_id=f.id, confidence=0.9, label=label)
    vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    session.add(
        DetectionEmbedding(
            id=str(uuid.uuid4()),
            detection_id=d.id,
            embedding_model_id="DINOV2-VITB14",
            vector=vector.astype(np.float16).tobytes(),
            dimension=4,
            l2_norm=1.0,
        )
    )
    session.flush()
    return d


def _canis_and_dogs(session, deployment_id: str, *, canis: int, dogs: int):
    """`canis` genus-level detections among `dogs` species-level ones.

    With every vector identical, each canis detection's neighbours are
    mostly 'domestic dog', which is a descendant of 'canis' and so a
    valid promotion suggestion. The dogs suggest their own label, which
    collapses to no suggestion, so they never join a cohort.
    """
    session.add_all(
        [
            LabelTaxonomy(
                id=str(uuid.uuid4()),
                classification_model_id=CLS_MODEL,
                name="canis",
                level="genus",
                taxon_class="mammalia",
                taxon_order="carnivora",
                taxon_family="canidae",
                taxon_genus="canis",
            ),
            LabelTaxonomy(
                id=str(uuid.uuid4()),
                classification_model_id=CLS_MODEL,
                name="domestic dog",
                level="species",
                taxon_class="mammalia",
                taxon_order="carnivora",
                taxon_family="canidae",
                taxon_genus="canis",
                taxon_species="familiaris",
            ),
        ]
    )
    made = []
    day = 1
    for _ in range(canis):
        made.append(
            _embedded_detection(
                session, deployment_id, label="canis",
                captured_at=datetime(2024, 1, day, 12),
            )
        )
        day += 1
    for _ in range(dogs):
        made.append(
            _embedded_detection(
                session, deployment_id, label="domestic dog",
                captured_at=datetime(2024, 1, day, 12),
            )
        )
        day += 1
    return made


# ── suggestions: the regression ──────────────────────────────────────────


def test_suggestions_under_cap_reports_the_full_pool_as_loaded(
    sort_db, fake_faiss
):
    """The bug: a narrowed result must not read as a capped one.

    Three canis detections form the only cohort, out of nine embedded
    detections that all fit under the cap. `total_detections` is 3 and
    `total_loaded` is 9, so the notice's test (`total_matching >
    total_loaded`) stays false while the old one (`> total_detections`)
    would have fired.
    """
    db_path, s = sort_db
    p = make_project(s, classification_model_id=CLS_MODEL)
    dep = make_deployment(s, project_id=p.id)
    _canis_and_dogs(s, dep.id, canis=3, dogs=6)
    s.commit()

    result = do_sort(
        db_path, p.id, {"sort": "suggestions", "filters": {}, "min_count": 2}
    )

    assert result["total_detections"] == 3   # the cohort
    assert result["total_loaded"] == 9       # everything the run read
    assert result["total_matching"] == 9     # everything that matched
    assert result["total_matching"] == result["total_loaded"]  # not capped
    labels = {d["label"] for d in result["detections"]}
    assert labels == {"canis"}


def test_suggestions_over_cap_reports_the_cap(sort_db, fake_faiss):
    """A genuinely capped suggestions run still says so."""
    db_path, s = sort_db
    p = make_project(s, classification_model_id=CLS_MODEL)
    dep = make_deployment(s, project_id=p.id)
    _canis_and_dogs(s, dep.id, canis=3, dogs=6)
    s.commit()

    result = do_sort(
        db_path,
        p.id,
        {
            "sort": "suggestions",
            "filters": {},
            "min_count": 2,
            "max_embeddings": 5,
        },
    )

    assert result["total_loaded"] == 5
    assert result["total_matching"] == 9
    assert result["total_matching"] > result["total_loaded"]  # capped


def test_suggestions_with_no_cohort_reports_the_pool_it_searched(
    sort_db, fake_faiss
):
    """No cohort is not a capped result either.

    Every detection already carries the species label, so nothing has a
    promotion to suggest. The grid is empty and the counts still have to
    describe the pool that was searched.
    """
    db_path, s = sort_db
    p = make_project(s, classification_model_id=CLS_MODEL)
    dep = make_deployment(s, project_id=p.id)
    _canis_and_dogs(s, dep.id, canis=0, dogs=6)
    s.commit()

    result = do_sort(
        db_path, p.id, {"sort": "suggestions", "filters": {}, "min_count": 2}
    )

    assert result["detections"] == []
    assert result["total_detections"] == 0
    assert result["total_loaded"] == 6
    assert result["total_matching"] == 6


# ── the sorts that return everything they load ───────────────────────────


def test_similarity_returns_everything_it_loads(sort_db, fake_faiss):
    db_path, s = sort_db
    p = make_project(s, classification_model_id=CLS_MODEL)
    dep = make_deployment(s, project_id=p.id)
    _canis_and_dogs(s, dep.id, canis=3, dogs=6)
    s.commit()

    result = do_sort(db_path, p.id, {"sort": "similarity", "filters": {}})

    assert result["total_detections"] == 9
    assert result["total_loaded"] == 9
    assert result["total_matching"] == 9


def test_event_sort_reports_loaded_and_matching(sort_db):
    """The metadata path carries the same three counts (no FAISS here)."""
    db_path, s = sort_db
    p = make_project(s, classification_model_id=CLS_MODEL)
    dep = make_deployment(s, project_id=p.id)
    for day in range(1, 4):
        make_detection(
            s,
            file_id=make_file(
                s,
                deployment_id=dep.id,
                captured_at_local=datetime(2024, 1, day, 12),
                width_px=1920,
                height_px=1080,
            ).id,
            confidence=0.9,
        )
    s.commit()

    full = do_sort(db_path, p.id, {"sort": "events", "filters": {}})
    assert full["total_detections"] == 3
    assert full["total_loaded"] == 3
    assert full["total_matching"] == 3

    capped = do_sort(
        db_path, p.id, {"sort": "events", "filters": {}, "max_detections": 2}
    )
    assert capped["total_detections"] == 2
    assert capped["total_loaded"] == 2
    assert capped["total_matching"] == 3


def test_flagged_and_liked_ride_on_rows_and_filter_the_pool(sort_db):
    """The Counts triage marks reach the crop grid: `file_flagged` /
    `file_favorited` ride on every row for the corner badge cluster,
    and the matching filters narrow the pool at the file level."""
    db_path, s = sort_db
    p = make_project(s, classification_model_id=CLS_MODEL)
    dep = make_deployment(s, project_id=p.id)

    def det(day, **file_marks):
        f = make_file(
            s,
            deployment_id=dep.id,
            captured_at_local=datetime(2024, 1, day, 12),
            width_px=1920,
            height_px=1080,
        )
        for k, v in file_marks.items():
            setattr(f, k, v)
        make_detection(s, file_id=f.id, confidence=0.9, label="canis")
        return f

    plain = det(1)
    marked = det(2, flagged=True, favorited=True)
    s.commit()

    rows = do_sort(db_path, p.id, {"sort": "events", "filters": {}})
    by_file = {d["file_id"]: d for d in rows["detections"]}
    assert by_file[marked.id]["file_flagged"] is True
    assert by_file[marked.id]["file_favorited"] is True
    assert by_file[plain.id]["file_flagged"] is False
    assert by_file[plain.id]["file_favorited"] is False

    got = do_sort(
        db_path, p.id, {"sort": "events", "filters": {"flagged": "flagged"}}
    )
    assert [d["file_id"] for d in got["detections"]] == [marked.id]
    got = do_sort(
        db_path,
        p.id,
        {"sort": "events", "filters": {"favorited": "not_favorited"}},
    )
    assert [d["file_id"] for d in got["detections"]] == [plain.id]


def test_empty_result_carries_every_count(sort_db, fake_faiss):
    """Nothing matched: all three counts are present and zero."""
    db_path, s = sort_db
    p = make_project(s, classification_model_id=CLS_MODEL)
    make_deployment(s, project_id=p.id)
    s.commit()

    for mode in ("similarity", "events", "suggestions"):
        result = do_sort(db_path, p.id, {"sort": mode, "filters": {}})
        assert result["total_detections"] == 0, mode
        assert result["total_loaded"] == 0, mode
        assert result["total_matching"] == 0, mode
