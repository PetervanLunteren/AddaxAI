"""Chunk id lists to stay under SQLite's bound-parameter limit."""

from collections.abc import Iterable, Iterator

# Max ids per `IN (...)` clause. SQLite's SQLITE_MAX_VARIABLE_NUMBER is 999 on
# old builds, 32766 on newer. A list built into `IN (?, ?, ...)` expands to one
# bound parameter per element, so any id list that scales with row count
# (files, detections, events) must be chunked. 900 stays under even the old
# limit. See the "too many SQL variables" crashes on large beta-tester datasets.
SQL_VAR_CHUNK = 900


def iter_id_chunks(
    ids: Iterable[str], size: int = SQL_VAR_CHUNK
) -> Iterator[list[str]]:
    """Yield successive <= ``size`` slices of ``ids``. Empty input yields nothing."""
    ids = list(ids)
    for i in range(0, len(ids), size):
        yield ids[i : i + size]
