"""
Independent events: the unit every statistic counts.

The stored unit is the sequence, a short burst of files (see
services/sequence_clustering). Whether two sequences are one sighting of an
animal is decided here, when a statistic asks, never by grouping files:

- Per species, per camera of a deployment (`Sequence.camera_key`): a
  sequence joins the running event of its species when it starts no more
  than the project's independence interval after the previous one of that
  species ended. The rolling rule, as camtrapR's ``deltaTimeComparedTo =
  "lastRecord"``. Interval 0 makes every sequence its own event. A sequence
  without a capture time is an event of its own.
- The event's count is that of its highest sequence (the most individuals
  of the species in one sequence, MaxN logic, as Camelot, wildrtrax and the
  WildCo course do), never a sum over its sequences. Its sex, age and
  behaviour rows are the cohort rows of that same sequence, so count and
  demographics always belong together. On a tie the earliest sequence wins.
- The event starts when its first sequence starts.

`independent_events(db, project_id)` gives one row per SequenceObservation
row that speaks for an event: ``observation_rowid`` (that row's SQLite
rowid), ``event_id`` (an integer, unique per event) and
``event_start_local``. A statistic joins it with `speaks_for(ie)`, which
keeps only those count rows, and then reads labels, categories, cohorts and
`effective_count` as it always did, with the event start in place of the
sequence start. Counting distinct ``event_id`` counts events.

The answer comes from a cache when it is fresh (DEVELOPERS.md, "The
independent events cache"): calculating it takes seconds on a large
project, reading it costs about what a plain sum does. Database triggers
mark every camera a write touches as stale (`db/triggers.py`), and
`refresh` recalculates those cameras. When it cannot, the same query runs
live, so a number never depends on the cache, only its speed does.

Filters a statistic adds after the join (labels, sites, a date window on
the event start) never change how sequences chain, because chaining is per
species and per camera inside one deployment. A date window keeps an event
by its start, so an event never counts twice across two windows.

Groups made before sequences existed are 30-minute groups; to this rule
each is one long sequence, and since two of them are always more than the
interval apart, an old project's statistics stay what they were.
"""

from __future__ import annotations

from sqlalchemy import (
    FromClause,
    Integer,
    Select,
    and_,
    case,
    cast,
    delete,
    exists,
    func,
    insert,
    literal_column,
    or_,
    select,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models import (
    Deployment,
    Project,
    Sequence,
    independent_event_cache,
    independent_event_stale,
)
from app.models.sequence_observation import SequenceObservation

logger = get_logger(__name__)

# The implicit rowid of the count row a statistic is reading. The cache is
# keyed on it: a join on this integer costs nothing, one on the text id made
# statistics 3.5 times slower.
OBSERVATION_ROWID = literal_column("sequence_observations.rowid")

# How long a refresh waits for SQLite's write lock before it gives up and
# the statistic is calculated live. A job holds the lock for minutes.
_REFRESH_WAIT_MS = 250

_stale = independent_event_stale.c
_cache = independent_event_cache.c


def species_key(obs: type[SequenceObservation] = SequenceObservation):
    """What one species is to the independence rule: its taxonomy row, else
    its label, else its category (a person or vehicle row)."""
    return func.coalesce(obs.label_taxonomy_id, obs.label, obs.category)


def speaks_for(ie: FromClause):
    """The join condition that keeps the count rows speaking for an event."""
    return ie.c.observation_rowid == OBSERVATION_ROWID


def _ms(column):
    """A camera-local time as whole milliseconds. Whole numbers, so a gap of
    exactly the interval compares equal: julianday's float seconds came out
    a hair over 1800 for 30 minutes."""
    return cast(func.round(func.julianday(column) * 86_400_000.0), Integer)


def _stale_camera():
    """True for a `Sequence` whose camera is marked stale."""
    return exists().where(
        _stale.deployment_id == Sequence.deployment_id,
        _stale.camera_key == Sequence.camera_key,
    )


def _events(project_id: str, *, stale_only: bool) -> Select:
    """The rows that speak for an event, for the whole project or only its
    stale cameras. One query for the live answer and the cache fill, so
    there is one definition of an event."""
    o, s = SequenceObservation, Sequence
    key = species_key(o)

    # One row per sequence and species: its total count and its time span.
    # Partitions use integer rowids: they are sorted three times.
    per_sequence = (
        select(
            s.id.label("sequence_id"),
            literal_column("sequences.rowid").label("sid"),
            literal_column("deployments.rowid").label("did"),
            s.camera_key,
            key.label("k"),
            s.sequence_start_local.label("start_local"),
            _ms(s.sequence_start_local).label("t0"),
            _ms(s.sequence_end_local).label("t1"),
            (Project.independence_interval * 1000).label("interval"),
            func.sum(o.effective_count).label("total"),
        )
        .select_from(o)
        .join(s, s.id == o.sequence_id)
        .join(Deployment, Deployment.id == s.deployment_id)
        .join(Project, Project.id == Deployment.project_id)
        .where(Deployment.project_id == project_id)
        .group_by(s.id, key)
    )
    if stale_only:
        per_sequence = per_sequence.where(_stale_camera())
    per_sequence = per_sequence.cte("per_sequence")
    p = per_sequence.c
    species_at_camera = (p.did, p.camera_key, p.k)
    in_time = (p.t0, p.sid)  # dateless first; each its own event

    # When the species was last seen at that camera before this sequence:
    # the latest end so far, which also covers sequences that overlap.
    last_seen = select(
        per_sequence,
        func.max(p.t1)
        .over(partition_by=species_at_camera, order_by=in_time, rows=(None, -1))
        .label("last_end"),
    ).cte("last_seen")
    ls = last_seen.c
    starts_event = case(
        (
            or_(
                ls.interval == 0,
                ls.t0.is_(None),
                ls.last_end.is_(None),
                ls.t0 - ls.last_end > ls.interval,
            ),
            1,
        ),
        else_=0,
    )
    numbered = select(
        last_seen,
        func.sum(starts_event)
        .over(
            partition_by=(ls.did, ls.camera_key, ls.k),
            order_by=(ls.t0, ls.sid),
            rows=(None, 0),
        )
        .label("event_no"),
    ).cte("numbered")
    n = numbered.c
    # One window for both: the highest sequence ranks first, and the start
    # is the earliest over the whole event.
    one_event = {
        "partition_by": (n.did, n.camera_key, n.k, n.event_no),
        "order_by": (n.total.desc(), n.t0, n.sid),
        "rows": (None, None),
    }
    ranked = select(
        n.sequence_id,
        n.k,
        func.row_number().over(**one_event).label("rank"),
        func.min(n.start_local).over(**one_event).label("event_start_local"),
    ).cte("ranked")
    r = ranked.c
    return (
        select(
            OBSERVATION_ROWID.label("observation_rowid"),
            # The event's rows are the highest sequence's rows of the
            # species; the lowest of their rowids names the event.
            func.min(OBSERVATION_ROWID)
            .over(partition_by=(r.sequence_id, r.k))
            .label("event_id"),
            r.event_start_local,
        )
        .select_from(o)
        .join(ranked, and_(r.sequence_id == o.sequence_id, r.k == key))
        .where(r.rank == 1)
    )


def _has_stale(db: Session, project_id: str) -> bool:
    return bool(
        db.execute(
            select(
                exists()
                .where(_stale.deployment_id == Deployment.id)
                .where(Deployment.project_id == project_id)
            )
        ).scalar()
    )


def refresh(db: Session, project_id: str) -> None:
    """Recalculate the cache of the project's stale cameras, if it can.

    Commits on the caller's session, so it only runs when that session has
    nothing of its own: an export or an import calls the statistics inside
    a long transaction, and a commit or rollback there would split or lose
    the job's work. Waits at most `_REFRESH_WAIT_MS` for the write lock.
    Either way it is left stale and `independent_events` answers live.
    """
    if not _has_stale(db, project_id):
        return
    conn = db.connection()
    if db.new or db.dirty or db.deleted or conn.connection.dbapi_connection.in_transaction:
        return

    from app.db.base import BUSY_TIMEOUT_S

    project_deployments = select(Deployment.id).where(Deployment.project_id == project_id)
    stale_rowids = (
        select(OBSERVATION_ROWID)
        .select_from(SequenceObservation)
        .join(Sequence, Sequence.id == SequenceObservation.sequence_id)
        .where(Sequence.deployment_id.in_(project_deployments))
        .where(_stale_camera())
    )
    conn.exec_driver_sql(f"PRAGMA busy_timeout = {_REFRESH_WAIT_MS}")
    try:
        try:
            # Stale rows of deleted deployments: nothing to recalculate.
            db.execute(
                delete(independent_event_stale).where(
                    _stale.deployment_id.not_in(select(Deployment.id))
                )
            )
            db.execute(
                delete(independent_event_cache).where(
                    _cache.observation_rowid.in_(stale_rowids)
                )
            )
            # In rowid order: the table then lies in the order statistics
            # read it, which kept a sum on 189K rows 40% slower otherwise.
            db.execute(
                insert(independent_event_cache).from_select(
                    ["observation_rowid", "event_id", "event_start_local"],
                    _events(project_id, stale_only=True).order_by(OBSERVATION_ROWID),
                )
            )
            db.execute(
                delete(independent_event_stale).where(
                    _stale.deployment_id.in_(project_deployments)
                )
            )
        finally:
            conn.exec_driver_sql(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_S * 1000)}")
    except OperationalError as e:
        db.rollback()
        logger.info(f"Independent events cache of {project_id} not refreshed: {e}")
        return
    db.commit()


def independent_events(db: Session, project_id: str) -> FromClause:
    """The count rows that speak for an independent event in the project,
    with the event they belong to: from the cache when it is fresh,
    calculated live otherwise. See the module docstring."""
    refresh(db, project_id)
    if _has_stale(db, project_id):
        return _events(project_id, stale_only=False).subquery("independent_events")
    return independent_event_cache
