"""
The independent events cache: which count rows speak for an event.

Derived data, never edited by hand and safe to empty at any time: the
statistics fall back to calculating events live, and the next request
fills the cache again. See `api/crud/independent_events.py` for the rule
and DEVELOPERS.md, "The independent events cache", for how it stays fresh.

- `independent_event_cache`: one row per `sequence_observations` row that
  speaks for an independent event. Keyed on that row's SQLite rowid, not its
  text id: a join on the integer is as fast as reading the table itself,
  the text id made statistics 3.5 times slower. `event_id` is the lowest
  rowid among the event's rows, so it is unique per event.
- `independent_event_stale`: the cameras (deployment plus
  `Sequence.camera_key`) whose cache rows are out of date. Database triggers
  (`db/triggers.py`) add a camera on every write that can change its
  events; a refresh recalculates those cameras and removes them.

Neither table has a foreign key on purpose. A cache row must be removable
by a trigger after its count row is gone, and a stale row is written by a
trigger while its deployment may be in the middle of a cascade delete.
"""

from sqlalchemy import Column, DateTime, Integer, String, Table

from app.db.base import Base

independent_event_cache = Table(
    "independent_event_cache",
    Base.metadata,
    Column("observation_rowid", Integer, primary_key=True, autoincrement=False),
    Column("event_id", Integer, nullable=False),
    Column("event_start_local", DateTime, nullable=True),
)

independent_event_stale = Table(
    "independent_event_stale",
    Base.metadata,
    Column("deployment_id", String(36), primary_key=True),
    Column("camera_key", String, primary_key=True),
)
