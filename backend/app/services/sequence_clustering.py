"""
Shared sequence-clustering primitive.

One source of truth for how AddaxAI groups files into sequences. Both
`generate_sequences_for_project` (which writes Sequence rows for the UI) and
`build_smoother_input` (which packages the same groupings for the
MegaDetector smoothing subprocess) call `cluster_files_into_sequences` here.

Rule: bucket files by folder, then within each folder start a new sequence
whenever the next file starts more than `SEQUENCE_GAP_SECONDS` after the
sequence so far ended. A video ends at its start plus its length; a photo
has no length. Folder bucketing keeps sequences from bridging across SD
cards when a user runs a mixed backlog as a single deployment.

The gap is short and fixed on purpose: a sequence is the burst a camera
records for one trigger or a quick run of them, cut by timestamps alone,
so no threshold, label or setting can move it (the stored unit people
confirm counts on must be stable). 120 s, as Agouti and WildTrax use; a
5-minute time-lapse cadence never bridges it. Whether two sequences are
one sighting of an animal is the independence interval, applied by the
statistics, not here. See DEVELOPERS.md, "Sequences and independent events".

Paired cameras (`Deployment.paired_cameras`) invert the folder rule: the
subfolders are dependent cameras, so all files go in one bucket and an
animal seen by more than one camera lands in one sequence. The caller
passes the flag; this module never reads the deployment itself.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta

from app.models import File

SEQUENCE_GAP_SECONDS = 120


def folder_key(f: File) -> str:
    """
    Return the clustering folder for a file: its path up to and including
    the last separator, `/` or `\\` whatever the OS. A plain string rule,
    so the migration that filled `Sequence.camera_key` for existing rows
    computes the same key in SQL. Post-2026-05 there are only `image` and
    `video` File rows, both at the camera's actual location on disk.
    """
    path = f.file_path
    return path[: max(path.rfind("/"), path.rfind("\\")) + 1]


def camera_key(f: File, *, paired_cameras: bool) -> str:
    """Which camera of its deployment a file belongs to: its folder, or ""
    when the deployment's subfolders are paired cameras (one camera). A
    sequence never spans two cameras, so it stores this for its first file
    (`Sequence.camera_key`), and the statistics apply the independence
    interval per camera with it."""
    return "" if paired_cameras else folder_key(f)


def file_end(f: File) -> datetime:
    """When a dated file ends: a video at its start plus its length, a photo
    (or a video whose length is unknown) at its start."""
    assert f.captured_at_local is not None
    return f.captured_at_local + timedelta(seconds=f.duration_seconds or 0)


def cluster_files_into_sequences(
    files: Iterable[File],
    *,
    paired_cameras: bool,
) -> list[list[File]]:
    """
    Partition `files` into sequence clusters.

    Each returned list is one sequence's files in capture-time order. A
    new sequence starts where a file starts more than
    `SEQUENCE_GAP_SECONDS` after every earlier file of the bucket ended.
    With `paired_cameras` off, clusters never span different folders
    either; with it on, the deployment's subfolders are one camera and
    files cluster across them. The flag has no default on purpose: every
    caller states which deployment semantics it wants.

    Files with `captured_at_local == None` (no EXIF capture date) can't
    be time-grouped, so each becomes its own single-file sequence at the
    end (with NULL sequence bounds). The caller is expected to filter by
    deployment first: this function does not check deployment membership
    and will cluster across deployments if given mixed input.
    """
    by_folder: dict[str, list[File]] = defaultdict(list)
    dateless: list[File] = []
    for f in files:
        if f.captured_at_local is None:
            dateless.append(f)
            continue
        by_folder[camera_key(f, paired_cameras=paired_cameras)].append(f)

    gap = timedelta(seconds=SEQUENCE_GAP_SECONDS)
    clusters: list[list[File]] = []
    # Iterate folders in a deterministic order so generated sequences and
    # smoother inputs are stable across runs (tests, logs, diffing).
    for key in sorted(by_folder):
        folder_files = sorted(by_folder[key], key=lambda f: f.captured_at_local)
        current: list[File] = [folder_files[0]]
        # The latest end so far, not the previous file's: with paired
        # cameras a long clip on one camera can outlast a photo on the other.
        end = file_end(folder_files[0])
        for f in folder_files[1:]:
            if f.captured_at_local - end > gap:
                clusters.append(current)
                current = [f]
                end = file_end(f)
            else:
                current.append(f)
                end = max(end, file_end(f))
        clusters.append(current)

    # Date-less files can't be time-grouped, so each is its own
    # single-file sequence (NULL sequence bounds). Keeps them visible in the
    # observation / taxa views; time-based stats exclude null-bound
    # sequences. Path-sorted for stable output.
    clusters.extend([f] for f in sorted(dateless, key=lambda f: f.file_path))

    return clusters
