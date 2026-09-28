"""Folder-run column policy for the shared export tables.

``tables_csv`` and ``tables_xlsx`` build their rows with the same
``export_crud`` builders that back the projects Export page, so the two
modes stay column-identical by default. A folder run has no sites, one
synthetic deployment, and no counts table, which leaves two of those
columns carrying nothing. This module trims them.

The trimming lives here, on the folder-run side, rather than in the
shared builders. Projects-mode exports therefore cannot regress by
construction: nothing in this file is reachable from them.

``deployment_id`` is dropped, and with it the rest of the site and
deployment part of the context block (``site_name``, ``latitude``,
``longitude``, ``site_tags``, ``deployment_start``, ``deployment_end``,
``deployment_tags``, ``trap_nights``). A folder run creates exactly one
queue entry, which becomes exactly one synthetic deployment (``rerun``
reuses it) with no site, so the id is the same UUID on every row and
points at a table the run never exports, and the other columns are
blank or repeat one value.

``notes`` is dropped. ``File.notes`` is writable over the API but no UI
ever sets it, so the column is always empty.

``n_events`` and ``n_individuals`` are dropped from the summary. Both are
ecological interpretation, which is what projects mode exists for: the
individuals column is the Counts table total, and a folder run has no
Counts table; the events column would be a headline "independent events"
figure computed from an interval almost no folder-run user sets. What is
left, images, videos and boxes per species, is what the AI directly
produced.

``event_id`` is kept, and so are ``event_start`` and ``event_end``: they
are the only columns that say which files belong to the same burst and
when it was. A key a user can group on is not a figure.
"""

from __future__ import annotations

from typing import Any

# Columns the shared builders emit that say nothing in a folder run.
OMITTED_COLUMNS = frozenset(
    {
        "site_name",
        "latitude",
        "longitude",
        "site_tags",
        "deployment_id",
        "deployment_start",
        "deployment_end",
        "deployment_tags",
        "trap_nights",
        "notes",
        "n_events",
        "n_individuals",
    }
)


def folder_run_table(
    headers: list[str],
    rows: list[list[Any]],
) -> tuple[list[str], list[list[Any]]]:
    """Return `(headers, rows)` without `OMITTED_COLUMNS`.

    Input is not mutated. Tables that carry none of the omitted columns
    pass through unchanged.
    """
    keep = [i for i, h in enumerate(headers) if h not in OMITTED_COLUMNS]
    return (
        [headers[i] for i in keep],
        [[row[i] for i in keep] for row in rows],
    )
