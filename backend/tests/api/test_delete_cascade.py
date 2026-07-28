"""Deleting analysis data: scope, SET NULL semantics, and cost.

The delete cascade is owned by the database (`ON DELETE CASCADE` plus
`passive_deletes=True` on the ORM relationships), not by SQLAlchemy walking
child collections in Python. These tests pin the three things that change
would break if it were ever undone: what gets deleted, what only gets
nulled, and how many statements it takes.

Background: re-running a large folder run used to take hours because the ORM
loaded every File, Detection and embedding into memory to delete them one by
one, while SQLite full-scanned `event_observations` for each deleted file
(its `max_n_file_id` FK had no index).
"""

import uuid

import numpy as np
from sqlalchemy import event as sa_event

from app.models import (
    Deployment,
    Detection,
    DetectionEmbedding,
    Event,
    EventObservation,
    File,
)
from app.models.event import event_files
from tests.conftest import (
    _engine,
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _make_run(db, *, files: int = 2):
    """A folder run with the full row chain hanging off it.

    Returns (project, deployment, files, event). Each file gets a detection
    with an embedding; the event holds one observation whose `max_n_file_id`
    points at the first file.
    """
    project = make_project(db, mode="folder_run")
    dep = make_deployment(
        db, project_id=project.id, folder_path=f"/fake/{uuid.uuid4().hex}"
    )
    made_files = []
    for _ in range(files):
        f = make_file(db, deployment_id=dep.id)
        det = make_detection(db, file_id=f.id)
        db.add(
            DetectionEmbedding(
                id=str(uuid.uuid4()),
                detection_id=det.id,
                embedding_model_id="TEST-EMB",
                vector=np.zeros(8, dtype=np.float16).tobytes(),
                dimension=8,
                l2_norm=0.0,
            )
        )
        made_files.append(f)

    ev = Event(id=str(uuid.uuid4()), deployment_id=dep.id, file_count=files)
    db.add(ev)
    db.flush()
    for seq, f in enumerate(made_files):
        db.execute(
            event_files.insert().values(
                event_id=ev.id, file_id=f.id, sequence_number=seq
            )
        )
    db.add(
        EventObservation(
            id=str(uuid.uuid4()),
            event_id=ev.id,
            label="deer",
            category="animal",
            max_n=1,
            max_n_file_id=made_files[0].id,
        )
    )
    db.commit()
    return project, dep, made_files, ev


def _counts(db, deployment_id: str) -> dict[str, int]:
    """Row counts across the whole chain below one deployment."""
    file_ids = [
        r[0]
        for r in db.query(File.id).filter(File.deployment_id == deployment_id)
    ]
    event_ids = [
        r[0]
        for r in db.query(Event.id).filter(
            Event.deployment_id == deployment_id
        )
    ]
    detection_ids = [
        r[0]
        for r in db.query(Detection.id).filter(Detection.file_id.in_(file_ids))
    ] if file_ids else []
    return {
        "deployments": db.query(Deployment)
        .filter(Deployment.id == deployment_id)
        .count(),
        "files": len(file_ids),
        "detections": len(detection_ids),
        "embeddings": db.query(DetectionEmbedding)
        .filter(DetectionEmbedding.detection_id.in_(detection_ids))
        .count()
        if detection_ids
        else 0,
        "events": len(event_ids),
        "event_files": db.query(event_files)
        .filter(event_files.c.event_id.in_(event_ids))
        .count()
        if event_ids
        else 0,
        "observations": db.query(EventObservation)
        .filter(EventObservation.event_id.in_(event_ids))
        .count()
        if event_ids
        else 0,
    }


def test_deleting_one_run_leaves_the_other_untouched(client, db):
    """Deleting run A wipes A's whole chain and touches nothing of run B.

    The `max_n_file_id` assertion is the important one: that column is an
    `ON DELETE SET NULL` FK to `files`, so a mis-scoped delete would quietly
    null out another run's observations instead of erroring.
    """
    project_a, dep_a, _, _ = _make_run(db, files=3)
    project_b, dep_b, _, _ = _make_run(db, files=3)

    before_b = _counts(db, dep_b.id)
    assert all(v > 0 for v in before_b.values())

    resp = client.delete(f"/api/folder-runs/{project_a.id}")
    assert resp.status_code == 204
    db.expire_all()

    after_a = _counts(db, dep_a.id)
    assert after_a == dict.fromkeys(after_a, 0), (
        f"run A left rows behind: {after_a}"
    )

    assert _counts(db, dep_b.id) == before_b

    # Run B's observation still points at run B's file.
    surviving = (
        db.query(EventObservation)
        .join(Event, Event.id == EventObservation.event_id)
        .filter(Event.deployment_id == dep_b.id)
        .all()
    )
    assert surviving
    assert all(o.max_n_file_id is not None for o in surviving)


def test_deleting_a_file_nulls_max_n_file_id_but_keeps_the_observation(db):
    """`max_n_file_id` is SET NULL, not CASCADE.

    An observation records a species count for an event; the file reference
    is only "the frame where the peak count was seen". Losing that frame
    must not lose the count.
    """
    _, dep, files, ev = _make_run(db, files=2)
    obs_id = (
        db.query(EventObservation.id)
        .filter(EventObservation.event_id == ev.id)
        .scalar()
    )

    db.delete(db.get(File, files[0].id))
    db.commit()
    db.expire_all()

    obs = db.get(EventObservation, obs_id)
    assert obs is not None, "observation was cascade-deleted, expected SET NULL"
    assert obs.max_n_file_id is None
    assert obs.max_n == 1


class _StatementCounter:
    """Count cursor executions on the shared test engine."""

    def __init__(self) -> None:
        self.count = 0

    def __enter__(self):
        sa_event.listen(_engine, "before_cursor_execute", self._on)
        return self

    def __exit__(self, *exc):
        sa_event.remove(_engine, "before_cursor_execute", self._on)

    def _on(self, conn, cursor, statement, params, context, executemany):
        self.count += 1


def test_deleting_a_deployment_costs_the_same_at_any_size(db):
    """Deleting a deployment must not scale with how much it holds.

    Before `passive_deletes=True` this emitted one SELECT per file, per
    detection and per event (277,014 statements for a 50k-file run). Now the
    database does the cascade, so the statement count is flat. Asserting the
    two sizes are *equal* pins O(1) without hard-coding a number that would
    churn on unrelated changes.
    """
    counts = []
    for n_files in (2, 20):
        _, dep, _, _ = _make_run(db, files=n_files)
        # Commit first so the collections are expired, matching a real
        # request where the session has not loaded the children. An
        # already-loaded collection still cascades in Python by design.
        db.commit()

        with _StatementCounter() as counter:
            db.delete(db.get(Deployment, dep.id))
            db.commit()
        counts.append(counter.count)

        assert db.query(Deployment).filter(Deployment.id == dep.id).count() == 0

    assert counts[0] == counts[1], (
        f"delete cost scales with size: {counts[0]} statements for 2 files "
        f"vs {counts[1]} for 20"
    )
