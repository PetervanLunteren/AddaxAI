"""
Shared event-clustering primitive.

One source of truth for how AddaxAI groups files into events. Both
`generate_events_for_project` (which writes Event rows for the UI) and
`build_smoother_input` (which packages the same groupings for the
MegaDetector smoothing subprocess) call `cluster_files_into_events` here.

Rule: bucket files by folder, then within each folder start a new event
whenever the time gap between consecutive files exceeds
`independence_interval`. Folder bucketing keeps events from bridging
across SD cards when a user runs a mixed backlog as a single
deployment. Time gaps handle the standard independence-interval rule
from camera-trap literature.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from app.models import File


def folder_key(f: File) -> str:
    """
    Return the clustering folder for a file: the file's own parent
    directory. Post-2026-05 there are only `image` and `video` File
    rows, both of which sit at the camera's actual location on disk, so
    no special-case rewrite is needed.
    """
    return str(Path(f.file_path).parent)


def cluster_files_into_events(
    files: Iterable[File],
    independence_interval: int,
) -> list[list[File]]:
    """
    Partition `files` into event clusters.

    Each returned list is one event's files in capture-time order.
    Clusters never span different folders and never contain two
    consecutive files whose capture-time gap exceeds
    `independence_interval` (seconds).

    Files with `captured_at_local == None` (no EXIF capture date) can't
    be time-grouped, so each becomes its own single-file event at the
    end (with NULL event bounds). The caller is expected to filter by
    deployment first — this function does not check deployment membership
    and will happily cluster across deployments if given mixed input.
    """
    by_folder: dict[str, list[File]] = defaultdict(list)
    dateless: list[File] = []
    for f in files:
        if f.captured_at_local is None:
            dateless.append(f)
            continue
        by_folder[folder_key(f)].append(f)

    clusters: list[list[File]] = []
    # Iterate folders in a deterministic order so generated events and
    # smoother inputs are stable across runs (tests, logs, diffing).
    for key in sorted(by_folder):
        folder_files = sorted(
            by_folder[key], key=lambda f: f.captured_at_local
        )
        current: list[File] = [folder_files[0]]
        for i in range(1, len(folder_files)):
            gap = (
                folder_files[i].captured_at_local
                - folder_files[i - 1].captured_at_local
            ).total_seconds()
            if gap > independence_interval:
                clusters.append(current)
                current = [folder_files[i]]
            else:
                current.append(folder_files[i])
        clusters.append(current)

    # Date-less files can't be time-grouped, so each is its own
    # single-file event (NULL event bounds). Keeps them visible in the
    # observation / taxa views; time-based stats exclude null-bound
    # events. Path-sorted for stable output.
    clusters.extend([f] for f in sorted(dateless, key=lambda f: f.file_path))

    return clusters
