"""The independent-events query: how stored sequences become the events
every statistic counts (app/api/crud/independent_events.py)."""

from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import func, select

from app.api.crud.independent_events import _events as _live
from app.api.crud.independent_events import independent_events, speaks_for
from app.models import independent_event_cache
from app.models.sequence import Sequence
from app.models.sequence_observation import SequenceObservation
from tests.conftest import make_deployment, make_project

T0 = datetime(2024, 6, 1, 12, 0, 0)


def _sequence(db, dep, start_min, end_min=None, *, camera="", rows=()):
    """A stored sequence from `start_min` to `end_min` minutes after T0 with
    observation rows ``(label, count[, sex])``. `start_min` None: dateless."""
    start = None if start_min is None else T0 + timedelta(minutes=start_min)
    end = None if start is None else T0 + timedelta(minutes=end_min or start_min)
    s = Sequence(
        deployment_id=dep.id, sequence_start_local=start, sequence_end_local=end,
        file_count=1, camera_key=camera,
    )
    db.add(s)
    db.flush()
    for label, count, *sex in rows:
        db.add(SequenceObservation(
            sequence_id=s.id, label=label, category="animal", max_n=count,
            sex=sex[0] if sex else None,
        ))
    db.flush()
    return s


def _speaking_rows(db, ie, project_id):
    from app.models import Deployment

    return set(db.execute(
        select(ie.c.observation_rowid, ie.c.event_id, ie.c.event_start_local)
        .select_from(SequenceObservation)
        .join(ie, speaks_for(ie))
        .join(Sequence, Sequence.id == SequenceObservation.sequence_id)
        .join(Deployment, Deployment.id == Sequence.deployment_id)
        .where(Deployment.project_id == project_id)
    ).all())


def _events(db, project_id):
    """{label: [(event start, count, [sexes]), ...]} in time order.

    Commits first, so the answer comes from the cache, and checks it is
    exactly the live answer."""
    db.commit()
    ie = independent_events(db, project_id)
    assert ie is independent_event_cache, "a clean session must get the cache"
    live = _live(project_id, stale_only=False).subquery()
    assert _speaking_rows(db, ie, project_id) == _speaking_rows(db, live, project_id)
    rows = db.execute(
        select(
            ie.c.event_id, ie.c.event_start_local, SequenceObservation.label,
            SequenceObservation.effective_count, SequenceObservation.sex,
        ).select_from(SequenceObservation).join(ie, speaks_for(ie))
    ).all()
    by_event = defaultdict(lambda: [None, None, 0, []])
    for event_id, start, label, count, sex in rows:
        e = by_event[event_id]
        e[0], e[1] = label, start
        e[2] += count
        e[3].append(sex)
    out = defaultdict(list)
    for label, start, count, sexes in by_event.values():
        out[label].append((start, count, sorted(sexes, key=str)))
    return {k: sorted(v, key=lambda e: (e[0] is not None, e[0] or T0)) for k, v in out.items()}


def _setup(db, interval=1800):
    project = make_project(db, independence_interval=interval)
    return project, make_deployment(db, project_id=project.id)


def _minutes(m):
    return T0 + timedelta(minutes=m)


def test_sequences_within_the_interval_are_one_event_with_the_highest_count(db):
    """Nine sequences of 5 deer and one of 10, minutes apart: one event of 10."""
    project, dep = _setup(db)
    for i in range(9):
        _sequence(db, dep, i * 3, rows=[("deer", 5)])
    _sequence(db, dep, 30, rows=[("deer", 10)])
    assert _events(db, project.id) == {"deer": [(_minutes(0), 10, [None])]}


def test_the_gap_is_measured_from_the_end_and_is_inclusive(db):
    """30 minutes after the previous end joins; one second more splits."""
    project, dep = _setup(db)
    _sequence(db, dep, 0, 10, rows=[("fox", 1)])
    _sequence(db, dep, 40, rows=[("fox", 1)])  # exactly 30 min after 10
    _sequence(db, dep, 70 + 1 / 60, rows=[("fox", 1)])  # 30 min 1 s after 40
    assert [e[0] for e in _events(db, project.id)["fox"]] == [_minutes(0), _minutes(70 + 1 / 60)]


def test_the_rule_is_rolling(db):
    """Each sequence is compared with the previous one, so a chain can last
    longer than the interval."""
    project, dep = _setup(db, interval=600)
    for m in (0, 9, 18, 27, 36):
        _sequence(db, dep, m, rows=[("boar", 2)])
    assert _events(db, project.id) == {"boar": [(_minutes(0), 2, [None])]}


def test_interval_zero_makes_every_sequence_an_event(db):
    project, dep = _setup(db, interval=0)
    _sequence(db, dep, 0, rows=[("deer", 1)])
    _sequence(db, dep, 0, rows=[("deer", 2)])
    assert len(_events(db, project.id)["deer"]) == 2


def test_raising_and_lowering_the_interval_regroups_at_once(db):
    """Nothing stored changes; the next question gets the new answer."""
    project, dep = _setup(db, interval=1800)
    _sequence(db, dep, 0, rows=[("deer", 1)])
    _sequence(db, dep, 20, rows=[("deer", 3)])
    assert len(_events(db, project.id)["deer"]) == 1
    project.independence_interval = 600
    db.flush()
    assert len(_events(db, project.id)["deer"]) == 2
    project.independence_interval = 3600
    db.flush()
    assert _events(db, project.id)["deer"] == [(_minutes(0), 3, [None])]


def test_species_chain_on_their_own(db):
    """Another species in between neither joins nor breaks a chain, and a
    sequence with two species counts once for each."""
    project, dep = _setup(db, interval=600)
    _sequence(db, dep, 0, rows=[("deer", 1), ("fox", 1)])
    _sequence(db, dep, 8, rows=[("hare", 1)])
    _sequence(db, dep, 16, rows=[("deer", 2)])
    events = _events(db, project.id)
    assert events["deer"] == [(_minutes(0), 1, [None]), (_minutes(16), 2, [None])]
    assert len(events["fox"]) == 1 and len(events["hare"]) == 1


def test_cameras_never_share_an_event(db):
    project, dep = _setup(db)
    _sequence(db, dep, 0, camera="/d/cam_a/", rows=[("deer", 1)])
    _sequence(db, dep, 1, camera="/d/cam_b/", rows=[("deer", 1)])
    assert len(_events(db, project.id)["deer"]) == 2


def test_deployments_never_share_an_event(db):
    project, dep = _setup(db)
    other = make_deployment(db, project_id=project.id)
    _sequence(db, dep, 0, rows=[("deer", 1)])
    _sequence(db, other, 1, rows=[("deer", 1)])
    assert len(_events(db, project.id)["deer"]) == 2


def test_the_cohorts_of_the_highest_sequence_speak_for_the_event(db):
    """2 females, then 1 male and 2 females: the second has the higher total
    (3), so the event is that sequence's cohorts, summed: 3, not 5."""
    project, dep = _setup(db)
    _sequence(db, dep, 0, rows=[("deer", 2, "female")])
    _sequence(db, dep, 5, rows=[("deer", 1, "male"), ("deer", 2, "female")])
    assert _events(db, project.id)["deer"] == [(_minutes(0), 3, ["female", "male"])]


def test_on_a_tie_the_earliest_sequence_wins(db):
    project, dep = _setup(db)
    _sequence(db, dep, 0, rows=[("deer", 2, "female")])
    _sequence(db, dep, 5, rows=[("deer", 2, "male")])
    assert _events(db, project.id)["deer"] == [(_minutes(0), 2, ["female"])]


def test_a_human_count_is_the_count(db):
    project, dep = _setup(db)
    first = _sequence(db, dep, 0, rows=[("deer", 1)])
    _sequence(db, dep, 5, rows=[("deer", 2)])
    db.query(SequenceObservation).filter_by(sequence_id=first.id).one().human_count = 4
    db.flush()
    assert _events(db, project.id)["deer"] == [(_minutes(0), 4, [None])]


def test_a_dateless_sequence_is_an_event_of_its_own(db):
    project, dep = _setup(db)
    _sequence(db, dep, None, rows=[("deer", 1)])
    _sequence(db, dep, None, rows=[("deer", 1)])
    _sequence(db, dep, 0, rows=[("deer", 1)])
    assert [e[0] for e in _events(db, project.id)["deer"]] == [None, None, _minutes(0)]


def test_old_long_groups_stay_one_event_each(db):
    """Groups the 30-minute rule made, more than 30 minutes apart: each one
    event, as before sequences existed."""
    project, dep = _setup(db)
    _sequence(db, dep, 0, 25, rows=[("deer", 2)])
    _sequence(db, dep, 56, 80, rows=[("deer", 1)])
    assert _events(db, project.id)["deer"] == [
        (_minutes(0), 2, [None]), (_minutes(56), 1, [None]),
    ]


def test_another_project_is_not_counted(db):
    """Each project's events come from its own rows, cached or live."""
    from app.models import Deployment

    project, dep = _setup(db)
    other, other_dep = _setup(db)
    _sequence(db, dep, 0, rows=[("deer", 1)])
    _sequence(db, other_dep, 0, rows=[("deer", 1)])
    for pid in (other.id, project.id):
        ie = independent_events(db, pid)
        n = db.execute(
            select(func.count())
            .select_from(SequenceObservation)
            .join(ie, speaks_for(ie))
            .join(Sequence, Sequence.id == SequenceObservation.sequence_id)
            .join(Deployment, Deployment.id == Sequence.deployment_id)
            .where(Deployment.project_id == pid)
        ).scalar()
        assert n == 1
