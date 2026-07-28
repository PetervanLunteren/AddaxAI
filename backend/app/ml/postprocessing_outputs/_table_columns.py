"""Folder-run column policy for the shared export tables.

``tables_csv`` and ``tables_xlsx`` build their rows with the same
``export_crud`` builders that back the projects Export page, so the two
modes stay column-identical by default. A folder run has no sites, one
synthetic deployment, and no counts table, which leaves two of those
columns carrying nothing. This module trims them.

The trimming lives here, on the folder-run side, rather than in the
shared builders. Projects-mode exports therefore cannot regress by
construction: nothing in this file is reachable from them.

``deployment_id`` is dropped. A folder run creates exactly one queue
entry, which becomes exactly one synthetic deployment (``rerun`` reuses
it), so the column is the same UUID on every row and points at a table
the run never exports.

``notes`` is dropped. ``File.notes`` is writable over the API but no UI
ever sets it, so the column is always empty.

``event_id`` is kept: it is the only column that says which files belong
to the same burst.
"""

from __future__ import annotations

from typing import Any

# Columns the shared builders emit that say nothing in a folder run.
OMITTED_COLUMNS = frozenset({"deployment_id", "notes"})


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
