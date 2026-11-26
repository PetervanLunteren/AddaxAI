"""
Independent events: the unit every statistic counts.

The stored unit is the sequence, a short burst of files (see
services/sequence_clustering). Whether two sequences are one sighting of an
animal is decided here, when a statistic asks, never stored:

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

The query returns one row per SequenceObservation row that speaks for an
event: ``observation_id``, ``event_id`` (unique per event within one query)
and ``event_start_local``. A statistic joins it to SequenceObservation,
which keeps only those rows, and then reads labels, categories, cohorts and
`effective_count` as it always did, with the event start in place of the
sequence start. Counting distinct ``event_id`` counts events.

Filters a statistic adds after the join (labels, sites, a date window on
the event start) never change how sequences chain, because chaining is per
species and per camera inside one deployment. A date window keeps an event
by its start, so an event never counts twice across two windows.

Sequences regrouped before sequences existed are 30-minute groups; to this
rule each is one long sequence, and since two of them are always more than
the interval apart, an old project's statistics stay what they were.
"""

from __future__ import annotations

from sqlalchemy import Integer, Subquery, and_, case, cast, func, or_, select

from app.models import Deployment, Project, Sequence
from app.models.sequence_observation import SequenceObservation


def species_key(obs: type[SequenceObservation] = SequenceObservation):
    """What one species is to the independence rule: its taxonomy row, else
    its label, else its category (a person or vehicle row)."""
    return func.coalesce(obs.label_taxonomy_id, obs.label, obs.category)


def _ms(column):
    """A camera-local time as whole milliseconds. Whole numbers, so a gap of
    exactly the interval compares equal: julianday's float seconds came out
    a hair over 1800 for 30 minutes."""
    return cast(func.round(func.julianday(column) * 86_400_000.0), Integer)


def independent_events(project_id: str) -> Subquery:
    """The observation rows that speak for an independent event in the
    project, with the event they belong to. See the module docstring."""
    o, s = SequenceObservation, Sequence
    key = species_key(o)

    # One row per sequence and species: its total count and its time span.
    per_sequence = (
        select(
            s.id.label("sequence_id"),
            s.deployment_id,
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
        .cte("per_sequence")
    )
    p = per_sequence.c
    species_at_camera = (p.deployment_id, p.camera_key, p.k)
    in_time = (p.t0, p.sequence_id)  # dateless first; each its own event

    # When the species was last seen at that camera before this sequence:
    # the latest end so far, which also covers sequences that overlap.
    last_seen = (
        select(
            per_sequence,
            func.max(p.t1)
            .over(partition_by=species_at_camera, order_by=in_time, rows=(None, -1))
            .label("last_end"),
        )
        .cte("last_seen")
    )
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
    numbered = (
        select(
            last_seen,
            func.sum(starts_event)
            .over(
                partition_by=(ls.deployment_id, ls.camera_key, ls.k),
                order_by=(ls.t0, ls.sequence_id),
                rows=(None, 0),
            )
            .label("event_no"),
        )
        .cte("numbered")
    )
    n = numbered.c
    one_event = (n.deployment_id, n.camera_key, n.k, n.event_no)
    ranked = (
        select(
            n.sequence_id,
            n.k,
            func.row_number()
            .over(partition_by=one_event, order_by=(n.total.desc(), n.t0, n.sequence_id))
            .label("rank"),
            func.first_value(n.sequence_id)
            .over(partition_by=one_event, order_by=(n.t0, n.sequence_id))
            .label("first_sequence_id"),
            func.min(n.start_local).over(partition_by=one_event).label("event_start_local"),
        )
        .cte("ranked")
    )
    r = ranked.c
    return (
        select(
            o.id.label("observation_id"),
            r.first_sequence_id.op("||")(":").op("||")(r.k).label("event_id"),
            r.event_start_local,
        )
        .select_from(o)
        .join(ranked, and_(ranked.c.sequence_id == o.sequence_id, ranked.c.k == key))
        .where(r.rank == 1)
        .subquery("independent_events")
    )
