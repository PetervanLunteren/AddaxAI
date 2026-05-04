"""
Shared event-clustering primitive.

One source of truth for how AddaxAI groups files into events. Both
`generate_events_for_project` (which writes Event rows for the UI) and
`build_smoother_input` (which packages the same groupings for the
MegaDetector smoothing subprocess) call `cluster_files_into_events` here.

The Timelapse runner uses the JSON twin `cluster_entries_into_events`,
which applies the same rule to (relative_path, captured_at_local) tuples
read directly from a results.json plus EXIF — no DB rows needed.

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
from datetime import datetime
from pathlib import Path

from app.models import File


def folder_key(f: File) -> str:
    """
    Return the clustering folder for a file.

    For images and videos, it's the file's own parent directory. For
    extracted video frames (`file_type='frame'`), the frame itself lives
    inside `.addaxai/video_frames/...`, which is a pipeline artifact
    path — not where the camera actually was. Fall back to the source
    video's parent so frames of one video cluster with images shot at
    the same camera.

    A frame row without a source_video (shouldn't happen in healthy
    data) falls back to its own file_path, which at worst over-splits
    by one.
    """
    if f.file_type == "frame" and f.source_video is not None:
        return str(Path(f.source_video.file_path).parent)
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

    Files with `captured_at_local == None` are skipped (defensive;
    should never happen on healthy data). The caller is expected to
    filter by deployment first — this function does not check
    deployment membership and will happily cluster across deployments
    if given mixed input.
    """
    by_folder: dict[str, list[File]] = defaultdict(list)
    for f in files:
        if f.captured_at_local is None:
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

    return clusters


def cluster_entries_into_events(
    entries: Iterable[tuple[str, datetime]],
    independence_interval: int,
) -> list[list[tuple[str, datetime]]]:
    """
    DB-less twin of `cluster_files_into_events`.

    Takes `(relative_path, captured_at_local)` tuples and applies the
    same folder-bucket + time-gap rule. Used by the Timelapse runner,
    which reads timestamps directly from the JSON's EXIF block instead
    of from a `File` row.

    Entries with `captured_at_local is None` are skipped (defensive).
    """
    by_folder: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    for relative_path, captured_at in entries:
        if captured_at is None:
            continue
        folder = str(Path(relative_path).parent)
        by_folder[folder].append((relative_path, captured_at))

    clusters: list[list[tuple[str, datetime]]] = []
    for key in sorted(by_folder):
        folder_entries = sorted(by_folder[key], key=lambda e: e[1])
        current: list[tuple[str, datetime]] = [folder_entries[0]]
        for i in range(1, len(folder_entries)):
            gap = (
                folder_entries[i][1] - folder_entries[i - 1][1]
            ).total_seconds()
            if gap > independence_interval:
                clusters.append(current)
                current = [folder_entries[i]]
            else:
                current.append(folder_entries[i])
        clusters.append(current)

    return clusters
