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
from sqlalchemy import select as sa_select
from sqlalchemy import text as sa_text

from app.models import (
    Deployment,
    Detection,
    DetectionEmbedding,
    Event,
    EventObservation,
    File,
    Project,
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


def test_purging_a_project_removes_the_whole_chain(db):
    """The staged teardown must leave exactly what the plain cascade left.

    `delete_project` empties the leaf tables in bulk before deleting the
    project, because SQLite runs a foreign key action program per row per
    level and emptying the leaves first means those find nothing to do
    (124 s down to 39 s on a 400k-file project). That is a speed change
    only: this pins that it is not also a behaviour change.
    """
    from app.api.crud import project as crud_project

    doomed, doomed_dep, _, _ = _make_run(db, files=3)
    keeper, keeper_dep, _, _ = _make_run(db, files=3)
    # Read the ids now: touching an attribute of a deleted instance later
    # re-queries the row and raises ObjectDeletedError.
    doomed_id, doomed_dep_id = doomed.id, doomed_dep.id
    keeper_id, keeper_dep_id = keeper.id, keeper_dep.id

    before = _counts(db, keeper_dep_id)
    assert all(v > 0 for v in before.values())

    assert crud_project.delete_project(db, doomed_id) is True

    assert _counts(db, doomed_dep_id) == {k: 0 for k in before}
    assert _counts(db, keeper_dep_id) == before
    assert db.query(Deployment).filter(Deployment.id == doomed_dep_id).count() == 0
    assert db.query(Project).filter(Project.id == keeper_id).count() == 1
    assert db.query(Project).filter(Project.id == doomed_id).count() == 0

    orphans = db.execute(sa_text("PRAGMA foreign_key_check")).fetchall()
    assert orphans == []


def test_purge_reports_what_it_removed(db):
    """The per-stage row counts are what the delete log line carries.

    They are also the only numbers a progress display could ever show:
    the plain cascade happens inside one statement and reports nothing.
    """
    from app.api.crud.deployment import purge_deployment_data

    project, dep, _, _ = _make_run(db, files=3)
    removed = dict(
        purge_deployment_data(
            db, sa_select(Deployment.id).where(Deployment.project_id == project.id)
        )
    )
    db.commit()

    assert removed["files"] == 3
    assert removed["detections"] == 3
    assert removed["detection_embeddings"] == 3
    assert removed["events"] == 1
    assert removed["event_files"] == 3
    assert removed["event_observations"] == 1


def test_purge_empties_the_leaves_before_their_parents(db):
    """The order is the whole point, so pin it.

    Run it the other way round and every parent delete pays the foreign
    key action for children that are still there, which is the slow case
    this exists to avoid. Nothing else would catch a reordering: the end
    state is identical either way.
    """
    from app.api.crud.deployment import purge_deployment_data

    project, _, _, _ = _make_run(db, files=2)
    order = [
        table
        for table, _ in purge_deployment_data(
            db, sa_select(Deployment.id).where(Deployment.project_id == project.id)
        )
    ]
    db.commit()

    assert order.index("detection_embeddings") < order.index("detections")
    assert order.index("detections") < order.index("files")
    assert order.index("event_observations") < order.index("events")
    assert order.index("event_files") < order.index("events")
    assert order.index("events") < order.index("files")


def test_a_folder_that_cannot_be_cleaned_does_not_fail_the_delete(
    client, db, tmp_path, make_unreadable
):
    """A disconnected drive must not turn a finished delete into a 500.

    The rows are already committed by the time the on-disk `.addaxai`
    cleanup runs, so an OS error there used to report "Internal Server
    Error" for a project that was in fact gone, and skip the cleanup for
    every remaining deployment too. Camera trap folders live on external
    drives, so this is the ordinary case, not an exotic one.
    """
    folder = tmp_path / "deployment"
    project = make_project(db)
    make_deployment(db, project_id=project.id, folder_path=str(folder))
    db.commit()
    project_id = project.id

    artifacts = folder / ".addaxai" / "projects" / project_id
    artifacts.mkdir(parents=True)
    (artifacts / "results.json").write_text("{}")
    make_unreadable(artifacts)

    resp = client.delete(f"/api/projects/{project_id}")

    assert resp.status_code == 204
    assert db.query(Project).filter(Project.id == project_id).count() == 0
