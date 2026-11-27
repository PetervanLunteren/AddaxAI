"""
The triggers that keep the independent events cache honest.

The cache (`models/independent_event_cache.py`) says which count rows speak
for an independent event. Anything that writes count rows, sequences or the
project's interval can change that, and those writes come from a dozen
places, raw SQL included (a package import, the taxa list relink, foreign
key cascades). So the database itself notes every camera such a write
touches, in `independent_event_stale`, and the next statistics request
recalculates those cameras (`api/crud/independent_events.refresh`). No code
path can forget to do it, including ones written later.

A camera is a deployment plus `Sequence.camera_key`: events never span two,
so it is the smallest unit that can be recalculated on its own.

These are the only triggers in the database. They are defined here, once,
and created by `init_db` at every start rather than by a migration: a
migration that rebuilds a table drops its triggers silently, and a shipped
migration cannot be corrected later. `init_db` drops them before a pending
upgrade and creates them again after it. See DEVELOPERS.md, "The
independent events cache".
"""

from __future__ import annotations

from sqlalchemy import Connection

_MARK = "INSERT OR IGNORE INTO independent_event_stale (deployment_id, camera_key) "

TRIGGERS: dict[str, str] = {
    # A new count row: its camera's events may change.
    "ie_observation_insert": f"""
        CREATE TRIGGER ie_observation_insert AFTER INSERT ON sequence_observations
        BEGIN
            {_MARK}SELECT deployment_id, camera_key FROM sequences WHERE id = NEW.sequence_id;
        END""",
    # A changed species or count. Sex, age and behaviour are not here: they
    # are read live, the cache only says which rows speak for an event.
    "ie_observation_update": f"""
        CREATE TRIGGER ie_observation_update
        AFTER UPDATE OF sequence_id, label, label_taxonomy_id, category, max_n, human_count
        ON sequence_observations
        WHEN OLD.sequence_id IS NOT NEW.sequence_id
          OR OLD.label IS NOT NEW.label
          OR OLD.label_taxonomy_id IS NOT NEW.label_taxonomy_id
          OR OLD.category IS NOT NEW.category
          OR OLD.max_n IS NOT NEW.max_n
          OR OLD.human_count IS NOT NEW.human_count
        BEGIN
            {_MARK}SELECT deployment_id, camera_key FROM sequences
                WHERE id IN (OLD.sequence_id, NEW.sequence_id);
        END""",
    # A removed count row. Its cache row goes at once, so a rowid SQLite
    # hands out again can never meet a leftover.
    "ie_observation_delete": f"""
        CREATE TRIGGER ie_observation_delete AFTER DELETE ON sequence_observations
        BEGIN
            DELETE FROM independent_event_cache WHERE observation_rowid = OLD.rowid;
            {_MARK}SELECT deployment_id, camera_key FROM sequences WHERE id = OLD.sequence_id;
        END""",
    # Before, not after: a cascade removes the count rows once the sequence
    # is gone, so their own trigger can no longer look its camera up.
    "ie_sequence_delete": f"""
        CREATE TRIGGER ie_sequence_delete BEFORE DELETE ON sequences
        BEGIN
            {_MARK}VALUES (OLD.deployment_id, OLD.camera_key);
        END""",
    # A sequence that moved (a deployment split, a camera offset) or changed
    # its span: both the camera it left and the one it is in now.
    "ie_sequence_update": f"""
        CREATE TRIGGER ie_sequence_update
        AFTER UPDATE OF deployment_id, camera_key, sequence_start_local, sequence_end_local
        ON sequences
        WHEN OLD.deployment_id IS NOT NEW.deployment_id
          OR OLD.camera_key IS NOT NEW.camera_key
          OR OLD.sequence_start_local IS NOT NEW.sequence_start_local
          OR OLD.sequence_end_local IS NOT NEW.sequence_end_local
        BEGIN
            {_MARK}VALUES (OLD.deployment_id, OLD.camera_key);
            {_MARK}VALUES (NEW.deployment_id, NEW.camera_key);
        END""",
    # A new interval changes every event of the project.
    "ie_project_interval": f"""
        CREATE TRIGGER ie_project_interval AFTER UPDATE OF independence_interval ON projects
        WHEN OLD.independence_interval IS NOT NEW.independence_interval
        BEGIN
            {_MARK}SELECT DISTINCT s.deployment_id, s.camera_key
                FROM sequences s JOIN deployments d ON d.id = s.deployment_id
                WHERE d.project_id = NEW.id;
        END""",
}


def drop_event_triggers(conn: Connection) -> None:
    """Remove the triggers, if there are any."""
    for name in TRIGGERS:
        conn.exec_driver_sql(f"DROP TRIGGER IF EXISTS {name}")


def create_event_triggers(conn: Connection) -> None:
    """Create the triggers as defined here, replacing any older version."""
    drop_event_triggers(conn)
    for sql in TRIGGERS.values():
        conn.exec_driver_sql(sql)


def forget_event_cache(conn: Connection) -> None:
    """Empty the cache and mark every camera stale.

    After a migration: a migration may rebuild a table, which renumbers the
    rowids the cache is keyed on. Each project fills again on its next
    statistics request.
    """
    conn.exec_driver_sql("DELETE FROM independent_event_cache")
    conn.exec_driver_sql(f"{_MARK}SELECT DISTINCT deployment_id, camera_key FROM sequences")
