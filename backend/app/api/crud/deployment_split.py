"""
Split one deployment into N child deployments along the folder hierarchy.

Users occasionally kick off an analysis against a root folder that actually
contains several deployments-worth of subfolders. They end up with one
`Deployment` row covering data that should have been N rows, which breaks
the map, camtrap-dp export, and site filtering.

This module implements the split operation:
1. Compute target subfolders at a chosen descent depth (clamped per-branch).
2. Copy the parent's `.addaxai/projects/{project_id}/` artifacts (results.json
   slice, video_frames subtree) into each child subfolder.
3. Validate the copies on disk.
4. In a single DB transaction: create child `Deployment` rows, reassign files,
   reassign events (duplicating events that straddle multiple children), and
   delete the parent row.
5. Delete the parent's old `.addaxai` artifacts.

See DEVELOPERS.md for the related datetime / non-label / verified conventions
that this pipeline preserves unchanged.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.crud.deployment import (
    _delete_deployment_artifacts,
    get_deployment,
)
from app.api.schemas.deployment import (
    SplitPreviewResponse,
    SplitPreviewTarget,
)
from app.core.logging_config import get_logger
from app.models import (
    AuditLog,
    Deployment,
    DeploymentQueue,
    Event,
    EventObservation,
    File,
    Job,
)
from app.models.event import event_files
from app.utils.fs_hidden import mkdir_hidden_addaxai

logger = get_logger(__name__)


class SplitError(Exception):
    """Raised when a split request fails its preconditions or pipeline."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class _TargetBucket:
    """One prospective child deployment and the files that belong to it."""

    folder_path: Path
    name: str
    files: list[File] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Folder walking
# ---------------------------------------------------------------------------


def _iter_child_dirs(folder: Path) -> list[Path]:
    """
    Subfolders of `folder` ignoring hidden entries (names starting with `.`).

    Also skips AddaxAI artifact folders (`.addaxai`) and OS metadata
    (`.DS_Store`, `__MACOSX`). Returns an empty list on permission errors.
    """
    try:
        entries = list(folder.iterdir())
    except (OSError, PermissionError) as exc:
        logger.warning(f"Cannot iterate folder {folder}: {exc}")
        return []
    return sorted(
        p
        for p in entries
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name != "__MACOSX"
    )


def _bucket_files_by_child(
    folder: Path, files: list[File], children: list[Path]
) -> tuple[dict[Path, list[File]], list[File]]:
    """
    Partition `files` by which visible child of `folder` they fall under.

    Returns ({child: bucket}, unmatched). `unmatched` lists files that live
    directly in `folder` (no child prefix match) or inside a hidden / skipped
    subdirectory. When unmatched is non-empty, descending further would
    orphan those files — the caller should treat `folder` as a clamp point.
    """
    buckets: dict[Path, list[File]] = {c: [] for c in children}
    unmatched: list[File] = []
    for f in files:
        # Path arithmetic, not string prefixes: a string test needs the
        # platform separator appended, and a hardcoded "/" matched nothing
        # against the backslash paths Windows stores.
        file_path = Path(f.file_path)
        matched: Path | None = None
        for c in children:
            if file_path.is_relative_to(c):
                matched = c
                break
        if matched is None:
            unmatched.append(f)
        else:
            buckets[matched].append(f)
    return buckets, unmatched


def _descend(
    folder: Path, depth_remaining: int, files: list[File]
) -> list[_TargetBucket]:
    """
    File-aware descent: walk `folder` down by up to `depth_remaining` levels,
    clamping a branch to its current folder when going deeper would either
    run out of visible subfolders or orphan files that live directly at this
    level.

    `files` is the set of non-frame File rows known to live under `folder`.
    Frame rows are not passed here because their file_path lives inside
    `.addaxai/` and wouldn't match any visible subfolder. The caller
    reattaches frames to their source video's bucket afterwards.

    Empty subfolders contribute nothing (skipped silently). The returned
    _TargetBucket.files is the disjoint subset of `files` that falls under
    each target.
    """
    if depth_remaining <= 0 or not files:
        return [_TargetBucket(folder_path=folder, name=folder.name, files=files)]

    children = _iter_child_dirs(folder)
    if not children:
        return [_TargetBucket(folder_path=folder, name=folder.name, files=files)]

    buckets, unmatched = _bucket_files_by_child(folder, files, children)
    if unmatched:
        # At least one file lives directly here (or in a hidden subdir).
        # Can't descend without losing it — clamp this branch.
        return [_TargetBucket(folder_path=folder, name=folder.name, files=files)]

    result: list[_TargetBucket] = []
    for c in children:
        bucket = buckets[c]
        if not bucket:
            # Empty subfolder — skip.
            continue
        result.extend(_descend(c, depth_remaining - 1, bucket))
    return result


def _max_descent_depth(folder: Path, files: list[File]) -> int:
    """
    Depth of the deepest file-aware descent that still produces distinct
    subfolders. 0 means `folder` itself is the only reachable target
    (either no children at all, or files live directly here).
    """
    if not files:
        return 0
    children = _iter_child_dirs(folder)
    if not children:
        return 0
    buckets, unmatched = _bucket_files_by_child(folder, files, children)
    if unmatched:
        return 0
    best = 0
    for c in children:
        bucket = buckets[c]
        if not bucket:
            continue
        best = max(best, _max_descent_depth(c, bucket))
    return 1 + best


def _count_media(files: list[File]) -> tuple[int, int]:
    """Count image vs video files in a list. Frame rows (pipeline artifacts)
    are ignored — they mirror their source video and aren't media the user
    thinks of as part of the deployment."""
    images = sum(1 for f in files if f.file_type == "image")
    videos = sum(1 for f in files if f.file_type == "video")
    return images, videos


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def _find_blocking_activity(
    db: Session, deployment: Deployment
) -> str | None:
    """
    Return a human-readable reason the deployment cannot be split right now,
    or None if it's safe to proceed.

    Blocks on:
    - folder_status != "valid" (we need working filesystem access to slice
      .addaxai artifacts safely).
    - Any DeploymentQueue entry with pending/processing status whose
      folder_path matches the deployment's folder (covers the
      queued-but-not-yet-started case where queue.deployment_id is still
      NULL) or whose deployment_id equals this deployment.
    - Any pending/running Job of a type that touches deployment artifacts
      for this project: `deployment_analysis`, `postprocessing`,
      `re_embedding`.
    """
    if deployment.folder_status != "valid":
        return (
            "Deployment folder is marked needs_relink. Reconnect the "
            "folder before splitting."
        )

    # Queue entries that are still in flight for this deployment.
    queue_stmt = (
        select(DeploymentQueue.id)
        .where(DeploymentQueue.status.in_(("pending", "processing")))
        .where(
            (DeploymentQueue.deployment_id == deployment.id)
            | (
                (DeploymentQueue.project_id == deployment.project_id)
                & (DeploymentQueue.folder_path == deployment.folder_path)
            )
        )
    )
    if db.execute(queue_stmt).first() is not None:
        return (
            "An analysis for this deployment is queued or running. "
            "Wait for it to finish before splitting."
        )

    # Jobs whose payload references this deployment's project. We match on
    # the JSON-encoded payload rather than trying to parse the column in SQL
    # — SQLite JSON functions aren't uniformly available. The row count here
    # is small (active jobs only).
    active_jobs = db.execute(
        select(Job).where(
            Job.status.in_(("pending", "running")),
            Job.type.in_(
                ("deployment_analysis", "postprocessing", "re_embedding")
            ),
        )
    ).scalars()
    for job in active_jobs:
        payload = job.payload or {}
        if payload.get("project_id") == deployment.project_id:
            return (
                f"A {job.type} job is currently {job.status}. "
                "Wait for it to finish before splitting."
            )

    return None


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def build_split_preview(
    db: Session, deployment_id: str, depth: int
) -> SplitPreviewResponse:
    """
    Compute the preview payload for a given depth.

    Empty subfolders (zero image+video) are omitted from `targets`. The
    `blocked_reason` field surfaces any precondition failure so the frontend
    can disable OK and explain why.
    """
    deployment = get_deployment(db, deployment_id)
    if deployment is None:
        raise SplitError(
            f"Deployment {deployment_id} not found", status_code=404
        )

    if not deployment.folder_path:
        return SplitPreviewResponse(
            original_folder=None,
            depth=depth,
            max_depth=0,
            can_decrease=depth > 1,
            can_increase=False,
            targets=[],
            blocked_reason="Deployment has no folder_path set.",
        )

    root = Path(deployment.folder_path)
    blocked_reason = _find_blocking_activity(db, deployment)

    # Only image/video File rows after 2026-05; both live at the
    # camera's actual on-disk path and feed the walker directly.
    bucketable = list(deployment.files)
    max_depth = _max_descent_depth(root, bucketable)
    effective_depth = max(1, depth)

    buckets = [
        b
        for b in _descend(root, effective_depth, bucketable)
        if b.folder_path != root
    ]

    targets: list[SplitPreviewTarget] = []
    for b in buckets:
        images, videos = _count_media(b.files)
        if images + videos == 0:
            continue
        targets.append(
            SplitPreviewTarget(
                folder_path=str(b.folder_path),
                name=b.name,
                image_count=images,
                video_count=videos,
            )
        )

    # Intentionally no blocked_reason when targets <= 1. The preview just
    # shows what the current depth would produce (zero or one child). The
    # OK button is disabled client-side when there is nothing meaningful
    # to split; the user can step the depth up without a scary warning.

    return SplitPreviewResponse(
        original_folder=str(root),
        depth=effective_depth,
        max_depth=max_depth,
        can_decrease=effective_depth > 1,
        can_increase=effective_depth < max_depth,
        targets=targets,
        blocked_reason=blocked_reason,
    )


# ---------------------------------------------------------------------------
# .addaxai slicing
# ---------------------------------------------------------------------------


def _slice_results_json(
    parent_json: dict,
    parent_folder: Path,
    child_folder: Path,
    child_file_paths: set[str],
) -> dict:
    """
    Build a `results.json` for a child deployment by filtering the parent's
    image list and rewriting each `"file"` entry to be relative to the child
    folder.

    Every top-level key other than `"images"` is copied verbatim so
    classification_categories, detection_categories, info etc. survive.
    """
    child_json = {k: v for k, v in parent_json.items() if k != "images"}

    new_images: list[dict] = []
    for img in parent_json.get("images", []):
        rel = img.get("file")
        if not rel:
            continue
        abs_path = parent_folder / rel
        if str(abs_path) not in child_file_paths:
            continue
        new_img = dict(img)
        try:
            new_img["file"] = str(abs_path.relative_to(child_folder))
        except ValueError:
            # Shouldn't happen: file is in child_file_paths but not under
            # child_folder. Skip defensively.
            logger.warning(
                f"File {abs_path} listed for child {child_folder} but not "
                f"relative to it; skipping"
            )
            continue
        new_images.append(new_img)

    child_json["images"] = new_images
    return child_json


def _copy_addaxai_slice(
    parent_folder: Path,
    child: _TargetBucket,
    project_id: str,
    parent_results: dict,
) -> None:
    """
    Write the child's slice of `.addaxai/projects/{project_id}/` into the
    child subfolder. Creates the destination directory, writes `results.json`,
    and copies `video_frames/` subtrees for each video File in the bucket.

    Raises on any I/O error. Caller is responsible for cleaning up partial
    state when this happens.
    """
    child_addaxai = child.folder_path / ".addaxai" / "projects" / project_id
    mkdir_hidden_addaxai(child_addaxai)

    child_file_paths = {f.file_path for f in child.files}
    child_json = _slice_results_json(
        parent_results, parent_folder, child.folder_path, child_file_paths
    )

    with (child_addaxai / "results.json").open("w") as fh:
        json.dump(child_json, fh)

    parent_frames_root = (
        parent_folder / ".addaxai" / "projects" / project_id / "video_frames"
    )
    if not parent_frames_root.exists():
        return

    child_frames_root = child_addaxai / "video_frames"
    for f in child.files:
        # Copy the frame subtree for every video in the bucket, regardless
        # of whether a best-frame was selected — frame rows attached to
        # this video still need the on-disk tree at the child location.
        if f.file_type != "video":
            continue
        abs_video = Path(f.file_path)
        try:
            rel_from_parent = abs_video.relative_to(parent_folder)
            rel_from_child = abs_video.relative_to(child.folder_path)
        except ValueError:
            logger.warning(
                f"Video {abs_video} is not under both parent {parent_folder} "
                f"and child {child.folder_path}; skipping frame copy"
            )
            continue
        src_dir = parent_frames_root / rel_from_parent
        if not src_dir.exists():
            continue
        dst_dir = child_frames_root / rel_from_child
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dst_dir)


def _validate_child_artifacts(
    child: _TargetBucket, project_id: str
) -> None:
    """
    Sanity-check the child's copied artifacts.

    Rule: every File row in the bucket must appear at least once in the
    child's results.json. Duplicate or extra entries in the JSON are
    tolerated, because parent results.json files in the wild sometimes
    carry duplicates from re-runs (the DB deduplicates via the unique
    (file_path, deployment_id) constraint but the JSON accumulates). A
    missing File would be a real problem — that's data loss.

    Also verifies each video File's expected best_frame_path exists on
    disk after the frame-tree copy.
    """
    child_addaxai = child.folder_path / ".addaxai" / "projects" / project_id
    json_path = child_addaxai / "results.json"
    if not json_path.exists():
        raise SplitError(
            f"Child {child.name}: results.json missing after copy"
        )
    try:
        with json_path.open() as fh:
            child_json = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise SplitError(
            f"Child {child.name}: results.json could not be read ({exc})"
        ) from exc

    written_paths = {
        img["file"] for img in child_json.get("images", []) if img.get("file")
    }
    expected_paths: set[str] = set()
    for f in child.files:
        try:
            expected_paths.add(
                str(Path(f.file_path).relative_to(child.folder_path))
            )
        except ValueError:
            continue
    missing = expected_paths - written_paths
    if missing:
        raise SplitError(
            f"Child {child.name}: {len(missing)} file(s) missing from "
            "results.json slice"
        )
    extras = len(child_json.get("images", [])) - len(written_paths)
    if extras > 0:
        logger.info(
            f"Child {child.name}: results.json has {extras} duplicate "
            "entries (preserved verbatim from parent)"
        )

    frames_root = child_addaxai / "video_frames"
    for f in child.files:
        if f.best_frame_number is None:
            continue
        try:
            rel_from_child = Path(f.file_path).relative_to(child.folder_path)
        except ValueError:
            continue
        expected = (
            frames_root / rel_from_child / f"frame{f.best_frame_number:06d}.jpg"
        )
        if not expected.exists():
            raise SplitError(
                f"Child {child.name}: expected best frame {expected} "
                "missing after copy"
            )


def _remove_child_artifacts(children: list[_TargetBucket]) -> None:
    """Best-effort cleanup of partially written child `.addaxai` dirs."""
    for child in children:
        target = child.folder_path / ".addaxai"
        try:
            if target.exists():
                shutil.rmtree(target)
        except OSError as exc:
            logger.warning(
                f"Failed to clean up partial artifacts at {target}: {exc}"
            )


# ---------------------------------------------------------------------------
# DB transaction
# ---------------------------------------------------------------------------


def _child_date_bounds(
    files: list[File],
) -> tuple[datetime | None, datetime | None]:
    """Return (min, max) captured_at_local across `files`, or (None, None)."""
    if not files:
        return None, None
    timestamps = [f.captured_at_local for f in files if f.captured_at_local is not None]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _files_grouped_by_child(
    buckets: list[_TargetBucket],
) -> dict[str, _TargetBucket]:
    """Return {file_id: bucket} so we can look up which child a file will move to."""
    mapping: dict[str, _TargetBucket] = {}
    for bucket in buckets:
        for f in bucket.files:
            mapping[f.id] = bucket
    return mapping


def _rewrite_frame_path(
    old: str | None,
    parent_folder: Path,
    child_folder: Path,
    project_id: str,
) -> str | None:
    """
    Translate a path inside the parent's `.addaxai/video_frames/` tree to the
    child's. Handles both `File.best_frame_path` on video rows and
    `File.file_path` on frame rows. Old path:
      {parent}/.addaxai/projects/{pid}/video_frames/<rel_from_parent>/frame*.jpg
    New path:
      {child}/.addaxai/projects/{pid}/video_frames/<rel_from_child>/frame*.jpg

    Returns None if `old` doesn't look like the expected layout.
    """
    if not old:
        return None
    old_path = Path(old)
    parent_frames_root = (
        parent_folder / ".addaxai" / "projects" / project_id / "video_frames"
    )
    try:
        rel_under_frames = old_path.relative_to(parent_frames_root)
    except ValueError:
        return None
    # rel_under_frames = <rel_video_from_parent>/frame<N>.jpg
    frame_name = rel_under_frames.name
    rel_video_from_parent = rel_under_frames.parent
    abs_video = parent_folder / rel_video_from_parent
    try:
        rel_video_from_child = abs_video.relative_to(child_folder)
    except ValueError:
        return None
    return str(
        child_folder
        / ".addaxai"
        / "projects"
        / project_id
        / "video_frames"
        / rel_video_from_child
        / frame_name
    )


def _reassign_events(
    db: Session,
    parent: Deployment,
    child_deployments: dict[str, Deployment],
    file_to_bucket: dict[str, _TargetBucket],
) -> None:
    """
    For every event on the parent deployment, group its files by target child
    and either reassign the event (single-group case) or duplicate it into
    per-child events preserving observations and verified flags.

    Observations are duplicated as-is. When an observation's max_n_file_id
    points to a file that ended up in a different child, we null it out on
    the side where that file no longer lives (the other side keeps the
    pointer so the peak-frame display stays intact where possible).
    """
    events = list(parent.events)
    for event in events:
        groups: dict[str, list[File]] = {}
        event_file_list = list(event.files)
        for f in event_file_list:
            bucket = file_to_bucket.get(f.id)
            if bucket is None:
                continue
            groups.setdefault(str(bucket.folder_path), []).append(f)

        if not groups:
            # Event has no files or none that fall under any child. Drop it.
            db.delete(event)
            continue

        if len(groups) == 1:
            only_folder = next(iter(groups))
            # Use the relationship attribute (not the FK column) so the
            # event is removed from parent.events via back_populates sync.
            event.deployment = child_deployments[only_folder]
            continue

        # Straddle case: create one new Event per child, duplicate
        # observations, then delete the original (which cascades event_files
        # and observations via ondelete=CASCADE on their FKs).
        observations = list(event.observations)
        for folder_str, files_here in groups.items():
            child_dep = child_deployments[folder_str]
            here_ids = {f.id for f in files_here}
            timestamps = [
                f.captured_at_local
                for f in files_here
                if f.captured_at_local is not None
            ]
            new_event = Event(
                deployment_id=child_dep.id,
                event_start_local=(
                    min(timestamps) if timestamps else event.event_start_local
                ),
                event_end_local=(
                    max(timestamps) if timestamps else event.event_end_local
                ),
                file_count=len(files_here),
                # Observations are duplicated as-is, so the human's
                # confirmation still holds for each child event.
                confirmed=event.confirmed,
            )
            db.add(new_event)
            db.flush()

            for seq, f in enumerate(files_here):
                db.execute(
                    event_files.insert().values(
                        event_id=new_event.id,
                        file_id=f.id,
                        sequence_number=seq,
                    )
                )

            for obs in observations:
                max_n_file_id = obs.max_n_file_id
                if max_n_file_id is not None and max_n_file_id not in here_ids:
                    max_n_file_id = None
                db.add(
                    EventObservation(
                        event_id=new_event.id,
                        label=obs.label,
                        label_taxonomy_id=obs.label_taxonomy_id,
                        category=obs.category,
                        max_n=obs.max_n,
                        max_n_file_id=max_n_file_id,
                        # Carry the human count override so a split never
                        # silently drops a confirmed count.
                        human_count=obs.human_count,
                    )
                )

        db.delete(event)


def _audit_split(
    db: Session,
    parent: Deployment,
    children: list[Deployment],
    depth: int,
) -> None:
    """Write a single audit_log row describing the split."""
    db.add(
        AuditLog(
            entity_type="deployment",
            entity_id=parent.id,
            action="delete",
            changes={
                "operation": "split",
                "depth": depth,
                "parent_folder": parent.folder_path,
                "children": [
                    {"id": c.id, "folder_path": c.folder_path}
                    for c in children
                ],
            },
        )
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def split_deployment(
    db: Session, deployment_id: str, depth: int
) -> list[str]:
    """
    Run the full split pipeline and return the IDs of the new child deployments.

    Raises SplitError on any precondition failure or I/O problem. The DB
    transaction is either fully committed or fully rolled back; on rollback,
    any partial child `.addaxai` directories are also removed from disk.
    """
    if depth < 1:
        raise SplitError("Split depth must be >= 1")

    parent = get_deployment(db, deployment_id)
    if parent is None:
        raise SplitError(
            f"Deployment {deployment_id} not found", status_code=404
        )
    if not parent.folder_path:
        raise SplitError("Deployment has no folder_path to split")

    blocked = _find_blocking_activity(db, parent)
    if blocked is not None:
        raise SplitError(blocked, status_code=409)

    parent_folder = Path(parent.folder_path)
    project_id = parent.project_id

    # Only `image` and `video` File rows exist post-2026-05; both live at
    # their actual on-disk path and feed the file-aware descent directly.
    buckets = [
        b
        for b in _descend(parent_folder, depth, list(parent.files))
        if b.folder_path != parent_folder
    ]

    if len(buckets) < 2:
        raise SplitError(
            "Split produces fewer than 2 non-empty deployments. Try a "
            "different depth."
        )

    # --- Parent results.json (read once; used by copy step) ------------------
    parent_json_path = (
        parent_folder / ".addaxai" / "projects" / project_id / "results.json"
    )
    if parent_json_path.exists():
        try:
            with parent_json_path.open() as fh:
                parent_results = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitError(
                f"Could not read parent results.json ({exc})"
            ) from exc
    else:
        # No results yet — the deployment existed but was never analysed.
        # Still allow the split; children will have no .addaxai artifacts.
        parent_results = None

    # --- Copy artifacts ------------------------------------------------------
    if parent_results is not None:
        try:
            for bucket in buckets:
                _copy_addaxai_slice(
                    parent_folder, bucket, project_id, parent_results
                )
            for bucket in buckets:
                _validate_child_artifacts(bucket, project_id)
        except Exception:
            _remove_child_artifacts(buckets)
            raise

    # --- DB transaction ------------------------------------------------------
    try:
        child_deployments: dict[str, Deployment] = {}
        for bucket in buckets:
            start, end = _child_date_bounds(bucket.files)
            child = Deployment(
                project_id=parent.project_id,
                site_id=parent.site_id,
                folder_path=str(bucket.folder_path),
                folder_status="valid",
                last_validated_at_utc=datetime.now(UTC),
                start_date_local=(
                    start.date() if start else parent.start_date_local
                ),
                end_date_local=end.date() if end else parent.end_date_local,
                camera_model=parent.camera_model,
                camera_serial=parent.camera_serial,
                notes=parent.notes,
                tags=dict(parent.tags or {}),
                datetime_offset_seconds=parent.datetime_offset_seconds,
            )
            db.add(child)
            db.flush()
            child_deployments[str(bucket.folder_path)] = child

        file_to_bucket = _files_grouped_by_child(buckets)
        for bucket in buckets:
            child = child_deployments[str(bucket.folder_path)]
            for f in bucket.files:
                # Use the relationship (not the FK column) so SQLAlchemy's
                # back_populates sync moves the File out of parent.files.
                # Otherwise db.delete(parent) below would cascade-delete it
                # via the cached collection.
                f.deployment = child
                if f.best_frame_path:
                    new_bfp = _rewrite_frame_path(
                        f.best_frame_path,
                        parent_folder,
                        bucket.folder_path,
                        project_id,
                    )
                    if new_bfp is not None:
                        f.best_frame_path = new_bfp
        db.flush()

        _reassign_events(
            db, parent, child_deployments, file_to_bucket
        )
        db.flush()

        children_for_audit = [
            child_deployments[str(b.folder_path)] for b in buckets
        ]
        _audit_split(db, parent, children_for_audit, depth)

        # DeploymentQueue.deployment_id is a plain string column with no FK
        # (deliberately, to avoid a circular table dependency). Null out any
        # queue rows that point at the parent so they don't dangle after the
        # delete below. The rest of the queue row is preserved for audit.
        from sqlalchemy import update
        db.execute(
            update(DeploymentQueue)
            .where(DeploymentQueue.deployment_id == parent.id)
            .values(deployment_id=None)
        )

        # Sanity: parent should now have no files and no events pointing at
        # it. Expire the collections so any lingering cache cannot cascade.
        db.expire(parent, ["files", "events"])
        db.delete(parent)
        db.commit()
    except Exception:
        db.rollback()
        if parent_results is not None:
            _remove_child_artifacts(buckets)
        raise

    # --- Post-commit: delete parent's old .addaxai ---------------------------
    # Same best-effort pattern used by delete_deployment: log but don't roll
    # back, because the DB swap has already landed and the worst case is a
    # stale folder sitting on disk.
    _delete_deployment_artifacts(str(parent_folder), project_id)

    created_ids = [
        child_deployments[str(b.folder_path)].id for b in buckets
    ]
    logger.info(
        f"Split deployment {deployment_id} into {len(created_ids)} "
        f"children at depth {depth}"
    )
    return created_ids


__all__ = [
    "SplitError",
    "build_split_preview",
    "split_deployment",
]
