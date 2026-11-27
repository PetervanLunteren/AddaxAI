"""The independent events cache stays equal to the live answer after every
kind of write (app/db/triggers.py, app/api/crud/independent_events.py)."""

from datetime import timedelta

import pytest
from sqlalchemy import func, select, text

from app.api.crud.independent_events import _events as _live
from app.api.crud.independent_events import independent_events, refresh
from app.api.crud.sequence_observation import (
    delete_sequence_observation,
    reset_sequence_to_ai,
    set_human_count,
    split_observation,
)
from app.models import (
    LabelTaxonomy,
    independent_event_cache,
    independent_event_stale,
)
from app.models.sequence_observation import SequenceObservation
from tests.api.test_independent_events import _sequence, _setup, _speaking_rows
from tests.conftest import make_deployment


def _assert_fresh(db, project_id):
    """Commit, read through the cache, and compare with the live answer."""
    db.commit()
    ie = independent_events(db, project_id)
    assert ie is independent_event_cache, "a clean session must get the cache"
    live = _live(project_id, stale_only=False).subquery()
    assert _speaking_rows(db, ie, project_id) == _speaking_rows(db, live, project_id)


@pytest.fixture()
def scene(db):
    """Two cameras of one deployment, a second deployment, deer and fox in
    sequences close enough to chain at a 10-minute interval, and the cache
    filled. Returns a namespace of what the writers need."""
    project, dep = _setup(db, interval=600)
    other = make_deployment(db, project_id=project.id)
    s = {
        "a1": _sequence(db, dep, 0, camera="/d/a/", rows=[("deer", 1), ("fox", 1)]),
        "a2": _sequence(db, dep, 5, camera="/d/a/", rows=[("deer", 3)]),
        "a3": _sequence(db, dep, 30, camera="/d/a/", rows=[("deer", 2)]),
        "b1": _sequence(db, dep, 1, camera="/d/b/", rows=[("deer", 1)]),
        "o1": _sequence(db, other, 0, rows=[("fox", 2)]),
    }
    _assert_fresh(db, project.id)
    assert db.execute(select(func.count()).select_from(independent_event_cache)).scalar()

    class Scene:
        pass

    sc = Scene()
    sc.project, sc.dep, sc.other, sc.seq = project, dep, other, s
    sc.obs = lambda key, label="deer": (
        db.query(SequenceObservation).filter_by(sequence_id=s[key].id, label=label).one()
    )
    return sc


def _taxonomy_row(db, sc):
    row = LabelTaxonomy(
        classification_model_id="TEST", name="deer", level="species",
        common_name="Deer", scientific_name="Cervus", is_custom=False,
    )
    db.add(row)
    db.flush()
    # Only a2's deer: its species key differs, so it chains on its own
    # until the row goes and the key falls back to the label.
    sc.obs("a2").label_taxonomy_id = row.id
    _assert_fresh(db, sc.project.id)
    return row


WRITERS = {
    "human count": lambda db, sc: set_human_count(db, sc.obs("a1").id, 9),
    "split a cohort": lambda db, sc: split_observation(db, sc.obs("a2").id),
    "delete a count row": lambda db, sc: delete_sequence_observation(
        db, split_observation(db, sc.obs("a2").id).id
    ),
    "reset to the AI": lambda db, sc: (
        set_human_count(db, sc.obs("a3").id, 7), db.commit(),
        reset_sequence_to_ai(db, sc.seq["a3"].id),
    ),
    "relabel a row": lambda db, sc: setattr(sc.obs("a2"), "label", "elk"),
    "bulk relabel in SQL": lambda db, sc: db.execute(
        text("UPDATE sequence_observations SET label = 'elk' WHERE sequence_id = :s"),
        {"s": sc.seq["a2"].id},
    ),
    "new count row": lambda db, sc: db.add(SequenceObservation(
        sequence_id=sc.seq["b1"].id, label="deer", category="animal", max_n=8,
    )),
    "delete a sequence": lambda db, sc: db.delete(sc.seq["a2"]),
    "delete a deployment": lambda db, sc: db.execute(
        text("DELETE FROM deployments WHERE id = :d"), {"d": sc.other.id}
    ),
    "move a sequence to another deployment": lambda db, sc: setattr(
        sc.seq["a3"], "deployment_id", sc.other.id
    ),
    "move a sequence to another camera": lambda db, sc: setattr(
        sc.seq["b1"], "camera_key", "/d/a/"
    ),
    "shift a camera's times in SQL": lambda db, sc: db.execute(
        text("UPDATE sequences SET sequence_start_local = datetime(sequence_start_local, "
             "'+1 hour'), sequence_end_local = datetime(sequence_end_local, '+1 hour') "
             "WHERE camera_key = '/d/a/' AND sequence_start_local > :t"),
        {"t": sc.seq["a1"].sequence_start_local + timedelta(minutes=1)},
    ),
    "change the interval": lambda db, sc: setattr(sc.project, "independence_interval", 3600),
    "delete a taxonomy row (set null)": lambda db, sc: db.delete(_taxonomy_row(db, sc)),
}


@pytest.mark.parametrize("writer", list(WRITERS), ids=list(WRITERS))
def test_the_cache_follows_every_kind_of_write(db, scene, writer):
    WRITERS[writer](db, scene)
    _assert_fresh(db, scene.project.id)


def test_a_refresh_recalculates_only_the_stale_cameras(db, scene):
    """Camera b's cache rows are removed by hand; an edit on camera a must
    not bring them back, because b was never marked stale."""
    db.execute(text(
        "DELETE FROM independent_event_cache WHERE observation_rowid IN ("
        " SELECT o.rowid FROM sequence_observations o JOIN sequences s"
        " ON s.id = o.sequence_id WHERE s.camera_key = '/d/b/')"
    ))
    set_human_count(db, scene.obs("a1").id, 4)
    db.commit()
    refresh(db, scene.project.id)
    in_b = db.execute(text(
        "SELECT count(*) FROM independent_event_cache c JOIN sequence_observations o"
        " ON o.rowid = c.observation_rowid JOIN sequences s ON s.id = o.sequence_id"
        " WHERE s.camera_key = '/d/b/'"
    )).scalar()
    assert in_b == 0
    assert db.execute(select(func.count()).select_from(independent_event_stale)).scalar() == 0


def test_a_session_with_pending_work_is_left_alone(db, scene):
    """An export calls the statistics inside its own transaction: the
    refresh must neither commit nor roll that back. It answers live."""
    scene.obs("a1").human_count = 6
    db.flush()
    ie = independent_events(db, scene.project.id)
    assert ie is not independent_event_cache
    assert db.connection().connection.dbapi_connection.in_transaction
    db.rollback()
    assert scene.obs("a1").human_count is None


def test_a_deleted_sequence_leaves_no_cache_row(db, scene):
    seq = scene.seq["a2"]
    rowids = [r for (r,) in db.execute(text(
        "SELECT rowid FROM sequence_observations WHERE sequence_id = :s"), {"s": seq.id})]
    db.delete(seq)
    db.commit()
    left = db.execute(
        select(func.count()).select_from(independent_event_cache)
        .where(independent_event_cache.c.observation_rowid.in_(rowids))
    ).scalar()
    assert left == 0
    stale = set(db.execute(select(independent_event_stale)).all())
    assert (scene.dep.id, "/d/a/") in stale


def test_only_sequence_rows_and_the_interval_are_watched(db, scene):
    """A sex or a note changes no event: no camera goes stale."""
    scene.obs("a1").sex = "female"
    scene.seq["a1"].notes = "foggy"
    db.commit()
    assert db.execute(select(func.count()).select_from(independent_event_stale)).scalar() == 0
