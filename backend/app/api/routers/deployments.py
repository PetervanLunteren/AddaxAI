"""
Deployment API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

import difflib
import io
import subprocess
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import deployment as crud_deployment
from app.api.crud import site as crud_site
from app.api.schemas.deployment import (
    BulkRelinkRequest,
    BulkRelinkResponse,
    BulkRelinkResultItem,
    DeploymentCreate,
    DeploymentResponse,
    DeploymentStatsOnly,
    DeploymentUpdate,
    DeploymentWithStats,
    FolderPreviewResponse,
    GPSCoordinates,
    GroupBrokenGroup,
    GroupBrokenItem,
    GroupBrokenRequest,
    GroupBrokenResponse,
    SampleFile,
    SuggestRelinkTargetRequest,
    SuggestRelinkTargetResponse,
)
from app.core.logging_config import get_logger
from app.core.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.db.base import get_db
from app.services.folder_scanner import scan_folder

logger = get_logger(__name__)
router = APIRouter(prefix="/api/deployments", tags=["Deployments"])


@router.get("", response_model=list[DeploymentResponse])
def list_deployments(
    site_id: str | None = Query(None, description="Filter by site ID"),
    project_id: str | None = Query(None, description="Filter by project ID"),
    db: Session = Depends(get_db),
) -> list[DeploymentResponse]:
    """
    List all deployments, optionally filtered by site_id or project_id.

    Returns empty list if no deployments exist.
    """
    deployments = crud_deployment.get_deployments(
        db, site_id=site_id, project_id=project_id
    )
    return [DeploymentResponse.model_validate(d) for d in deployments]


@router.post("", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
def create_deployment(
    deployment: DeploymentCreate, db: Session = Depends(get_db)
) -> DeploymentResponse:
    """
    Create a new deployment.

    Returns 400 if site_id is invalid (foreign key constraint).
    """
    try:
        db_deployment = crud_deployment.create_deployment(db, deployment)
        logger.info(f"Created deployment for site {deployment.site_id} (ID: {db_deployment.id})")
        return DeploymentResponse.model_validate(db_deployment)
    except IntegrityError as e:
        # Foreign key constraint violation (invalid site_id)
        logger.warning(f"Failed to create deployment: site {deployment.site_id} not found")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid site_id: {deployment.site_id}",
        ) from e


@router.get("/preview-folder", response_model=FolderPreviewResponse)
def preview_folder_path(
    path: str = Query(..., description="Absolute path to folder to preview"),
) -> FolderPreviewResponse:
    """
    Preview a folder before creating a deployment.

    Scans the folder to count images/videos and check for GPS coordinates.
    Used by the frontend to validate folder selection before adding to queue.

    Returns 400 if folder doesn't exist or isn't accessible.
    """
    if not path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder path is required",
        )

    # Scan folder
    try:
        logger.info(f"Scanning folder: {path}")
        preview = scan_folder(path)
        img_count = preview['image_count']
        vid_count = preview['video_count']
        logger.info(
            f"Folder scan complete: {img_count} images, "
            f"{vid_count} videos"
        )
    except FileNotFoundError as e:
        logger.error(f"Folder not found: {path}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Folder not found: {str(e)}",
        ) from e
    except PermissionError as e:
        logger.error(f"Permission denied: {path}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(f"Error scanning folder: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scanning folder: {str(e)}",
        ) from e

    # Convert to response schema
    return FolderPreviewResponse(
        image_count=preview["image_count"],
        video_count=preview["video_count"],
        total_count=preview["total_count"],
        gps_location=GPSCoordinates(**preview["gps_location"]) if preview["gps_location"] else None,
        suggested_site_id=None,
        sample_files=[SampleFile(**sf) for sf in preview["sample_files"]],
        start_date=preview["start_date"],
        end_date=preview["end_date"],
        missing_datetime=preview["missing_datetime"],
        datetime_validation_log=preview["datetime_validation_log"],
    )


# Maximum thumbnail width for preview images (avoids sending 20 MB RAW files)
_PREVIEW_MAX_WIDTH = 800


@router.get("/preview-image")
def preview_image(
    folder: str = Query(..., description="Absolute path to deployment folder"),
    file: str = Query(..., description="Relative file path from sample_files"),
):
    """Serve a resized image from a deployment folder for datetime preview.

    Used by the DatetimeOffsetModal to show sample images so users can
    compare the burned-in pixel date with the extracted EXIF datetime.

    Security: rejects paths containing '..' and validates the resolved
    path is inside the folder.
    """
    # Block path traversal
    if ".." in file or PurePosixPath(file).is_absolute():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )

    folder_path = Path(folder)
    file_path = (folder_path / file).resolve()

    # Ensure resolved path is inside the folder
    if not str(file_path).startswith(str(folder_path.resolve())):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File path is outside the deployment folder",
        )

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file}",
        )

    ext = file_path.suffix.lower()
    is_video = ext in VIDEO_EXTENSIONS

    try:
        if is_video:
            # Extract first frame from video via ffmpeg → JPEG pipe
            result = subprocess.run(
                [
                    "ffmpeg", "-i", str(file_path),
                    "-vframes", "1",
                    "-vf", f"scale={_PREVIEW_MAX_WIDTH}:-1",
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout:
                raise RuntimeError(
                    f"ffmpeg failed (exit {result.returncode}): "
                    f"{result.stderr[:200].decode(errors='replace')}"
                )
            buf = io.BytesIO(result.stdout)
            return StreamingResponse(buf, media_type="image/jpeg")

        # Image: open with Pillow, resize, serve as JPEG
        img = Image.open(file_path)
        if img.width > _PREVIEW_MAX_WIDTH:
            ratio = _PREVIEW_MAX_WIDTH / img.width
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Failed to serve preview for {file_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read file: {str(e)}",
        ) from e


@router.get("/bulk-stats", response_model=dict[str, DeploymentStatsOnly])
def get_bulk_stats(
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_db),
) -> dict[str, DeploymentStatsOnly]:
    """
    Get file/event/detection counts for all deployments in a project.

    Returns a dict keyed by deployment ID with stats for each.
    """
    raw = crud_deployment.get_bulk_deployment_stats(db, project_id)
    return {
        dep_id: DeploymentStatsOnly(**stats)
        for dep_id, stats in raw.items()
    }


@router.post("/bulk-relink", response_model=BulkRelinkResponse)
def bulk_relink_deployments(
    request: BulkRelinkRequest, db: Session = Depends(get_db)
) -> BulkRelinkResponse:
    """
    Relink multiple deployments to new folders in one request.

    Iterates the provided replacements and runs the same per-deployment
    relink logic (sample verification + bulk File.file_path rewrite) for
    each one. Per-item failures (verification mismatches, missing
    deployments) are reported in the results array; the operation as a
    whole always returns 200 — partial success is the expected case.
    """
    results: list[BulkRelinkResultItem] = []
    for item in request.replacements:
        relink_result = crud_deployment.relink_deployment(
            db, item.deployment_id, item.new_folder_path
        )
        mismatches = (
            relink_result.verify_result.mismatches
            if relink_result.verify_result
            else []
        )
        results.append(
            BulkRelinkResultItem(
                deployment_id=item.deployment_id,
                success=relink_result.success,
                files_rewritten=relink_result.files_rewritten,
                mismatches=mismatches,
            )
        )
    return BulkRelinkResponse(results=results)


# Similarity threshold above which we auto-suggest a sibling folder as
# a replacement. 0.6 catches common renames like `foo` → `foo1` or
# `foo_v2` while still rejecting completely unrelated siblings.
_SUGGEST_SIMILARITY_THRESHOLD = 0.6


@router.post("/suggest-relink-target", response_model=SuggestRelinkTargetResponse)
def suggest_relink_target(
    request: SuggestRelinkTargetRequest,
) -> SuggestRelinkTargetResponse:
    """
    Suggest a replacement folder for a missing deployment path.

    Walks up from `missing_path` until finding an ancestor that still
    exists on disk. The *deepest-missing-ancestor* (the direct child
    of that surviving parent, in the missing path) is what actually
    got renamed — so we compare siblings of the surviving parent to
    that name, not to the leaf of the missing path. If a close match
    is found, we reconstruct the full suggestion by appending the
    remaining sub-path.

    Example: missing=/a/b/old_name/c/d, surviving parent=/a/b,
    deepest-missing-ancestor=old_name, tail=c/d. If /a/b/new_name
    exists and is similar to old_name, the suggestion becomes
    /a/b/new_name/c/d.

    Used by the bulk relink banner to pre-fill the "new folder" field
    after a drive rename.
    """
    missing = Path(request.missing_path)
    chain = [missing, *missing.parents]

    # Walk up until we hit a surviving parent. Remember the index so
    # we can identify the deepest-missing-ancestor (one level below).
    surviving_index: int | None = None
    for i, candidate in enumerate(chain):
        if candidate.exists() and candidate.is_dir():
            surviving_index = i
            break

    if surviving_index is None:
        return SuggestRelinkTargetResponse()

    surviving_parent = chain[surviving_index]

    # If the missing path itself exists, there's nothing to suggest.
    if surviving_index == 0:
        return SuggestRelinkTargetResponse(existing_parent=str(surviving_parent))

    deepest_missing = chain[surviving_index - 1]
    missing_name = deepest_missing.name

    try:
        tail = missing.relative_to(deepest_missing)
    except ValueError:
        tail = Path("")

    try:
        siblings = [
            p for p in surviving_parent.iterdir() if p.is_dir() and p.name != missing_name
        ]
    except (PermissionError, OSError):
        return SuggestRelinkTargetResponse(existing_parent=str(surviving_parent))

    # Rank siblings by name similarity to the deepest-missing-ancestor.
    scored = sorted(
        (
            (
                difflib.SequenceMatcher(None, missing_name, p.name).ratio(),
                p,
            )
            for p in siblings
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )

    # Reconstruct full candidate paths by appending the tail. Prefer
    # paths that actually exist on disk so we don't suggest a dead end.
    candidates: list[str] = []
    suggested: str | None = None
    for score, sibling in scored[:10]:
        full = sibling / tail if str(tail) else sibling
        candidates.append(str(full))
        if (
            suggested is None
            and score >= _SUGGEST_SIMILARITY_THRESHOLD
            and full.exists()
            and full.is_dir()
        ):
            suggested = str(full)

    return SuggestRelinkTargetResponse(
        existing_parent=str(surviving_parent),
        suggested_path=suggested,
        candidates=candidates[:5],
    )


def _walk_to_surviving_parent(missing: Path) -> tuple[Path | None, Path | None]:
    """
    Walk up from `missing` to find the deepest existing ancestor.

    Returns (surviving_parent, deepest_missing_ancestor) where the latter
    is the direct child of surviving_parent in the missing chain — i.e.,
    the folder that actually got renamed/moved. Returns (None, None) if
    nothing in the chain exists, and (parent, None) if the missing path
    itself still exists (no fix needed).
    """
    chain = [missing, *missing.parents]
    for i, candidate in enumerate(chain):
        if candidate.exists() and candidate.is_dir():
            if i == 0:
                return candidate, None
            return candidate, chain[i - 1]
    return None, None


@router.post("/group-broken", response_model=GroupBrokenResponse)
def group_broken_deployments(
    request: GroupBrokenRequest,
) -> GroupBrokenResponse:
    """
    Group a list of broken deployments by the *deepest missing ancestor*
    they share, then auto-suggest a replacement folder per group.

    Why this is a backend endpoint rather than a frontend path-string
    grouping pass: only the filesystem knows where the actual problem
    sits. Two deployments under `/a/b/proj1/` and `/a/b/proj2/` look
    like separate problems by path, but if `/a/b` was the renamed
    folder, they're really the same problem and should fix together.

    Algorithm: for each item, walk up to find its surviving parent and
    deepest-missing-ancestor. Items that share the same
    (surviving_parent, deepest_missing) pair go in the same group. For
    each group, rank siblings of the surviving parent by name similarity
    to the deepest-missing-ancestor and return the best match (verified
    to exist on disk) as `suggested_path`.
    """
    # Bucket items by (surviving_parent, deepest_missing) pair.
    buckets: dict[tuple[str, str], list[GroupBrokenItem]] = {}
    fallback_groups: list[GroupBrokenGroup] = []

    for item in request.items:
        missing = Path(item.folder_path)
        surviving_parent, deepest_missing = _walk_to_surviving_parent(missing)

        if surviving_parent is None or deepest_missing is None:
            # Either nothing in the chain exists, or the path itself still
            # exists (caller shouldn't have included it). Emit a single-item
            # fallback group with no suggestion so the user can still pick
            # a folder manually.
            fallback_groups.append(
                GroupBrokenGroup(
                    prefix=str(missing),
                    existing_parent=(
                        str(surviving_parent) if surviving_parent else None
                    ),
                    suggested_path=None,
                    items=[item],
                )
            )
            continue

        key = (str(surviving_parent), str(deepest_missing))
        buckets.setdefault(key, []).append(item)

    groups: list[GroupBrokenGroup] = []
    for (existing_parent_str, deepest_missing_str), items in buckets.items():
        existing_parent = Path(existing_parent_str)
        deepest_missing = Path(deepest_missing_str)
        missing_name = deepest_missing.name

        try:
            siblings = [
                p
                for p in existing_parent.iterdir()
                if p.is_dir() and p.name != missing_name
            ]
        except (PermissionError, OSError):
            siblings = []

        scored = sorted(
            (
                (
                    difflib.SequenceMatcher(None, missing_name, p.name).ratio(),
                    p,
                )
                for p in siblings
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )

        # Pick the best sibling whose reconstructed path actually exists
        # for at least one sample item — avoids dead-end suggestions.
        suggested: str | None = None
        sample = items[0]
        sample_tail: Path
        try:
            sample_tail = Path(sample.folder_path).relative_to(deepest_missing)
        except ValueError:
            sample_tail = Path("")

        for score, sibling in scored:
            if score < _SUGGEST_SIMILARITY_THRESHOLD:
                break
            reconstructed = sibling / sample_tail if str(sample_tail) else sibling
            if reconstructed.exists() and reconstructed.is_dir():
                suggested = str(sibling)
                break

        groups.append(
            GroupBrokenGroup(
                prefix=str(deepest_missing),
                existing_parent=str(existing_parent),
                suggested_path=suggested,
                items=items,
            )
        )

    return GroupBrokenResponse(groups=groups + fallback_groups)


@router.get("/file-datetime")
def get_file_datetime(
    folder: str = Query(..., description="Absolute path to deployment folder"),
    file: str = Query(..., description="Relative file path"),
):
    """Extract the EXIF/metadata datetime from a single file on demand.

    Called lazily by the DatetimeOffsetModal as the user navigates through
    images, so we don't pay the cost of extracting all 10k+ datetimes
    during the initial folder scan.
    """
    if ".." in file or PurePosixPath(file).is_absolute():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )

    folder_path = Path(folder)
    file_path = (folder_path / file).resolve()

    if not str(file_path).startswith(str(folder_path.resolve())):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File path is outside the deployment folder",
        )

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file}",
        )

    dt = None
    ext = file_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        from app.services.folder_scanner import _extract_exif_date_single

        dt = _extract_exif_date_single(file_path)
    elif ext in VIDEO_EXTENSIONS:
        from app.utils.media_dates import extract_video_date

        dt = extract_video_date(file_path)

    return {"file_datetime": dt.isoformat() if dt else None}


@router.get("/{deployment_id}", response_model=DeploymentResponse)
def get_deployment(deployment_id: str, db: Session = Depends(get_db)) -> DeploymentResponse:
    """
    Get deployment by ID.

    Returns 404 if deployment doesn't exist.
    """
    db_deployment = crud_deployment.get_deployment(db, deployment_id)
    if db_deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )
    return DeploymentResponse.model_validate(db_deployment)


@router.patch("/{deployment_id}", response_model=DeploymentResponse)
def update_deployment(
    deployment_id: str, deployment: DeploymentUpdate, db: Session = Depends(get_db)
) -> DeploymentResponse:
    """
    Update an existing deployment.

    Returns 404 if deployment doesn't exist.
    Use this endpoint to re-link folder paths if files have moved,
    or to move a deployment to a different site in the same project.

    Cross-project moves are not allowed (returns 403).
    If folder_path changes, the backend runs sample-based verification
    against the new folder. If verification fails, returns 400 with the
    list of mismatches in the response body.
    """
    update_fields = deployment.model_dump(exclude_unset=True)

    # Fetch the current deployment once for all the pre-flight checks below.
    current = crud_deployment.get_deployment(db, deployment_id)
    if current is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )

    # If site_id is changing, validate the new site exists and
    # belongs to the same project as the current one.
    if "site_id" in update_fields and update_fields["site_id"] is not None:
        new_site_id = update_fields["site_id"]
        if new_site_id != current.site_id:
            new_site = crud_site.get_site(db, new_site_id)
            if new_site is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Site with id '{new_site_id}' not found",
                )
            if new_site.project_id != current.site.project_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot move deployment to a site in a different project",
                )

    # If folder_path is changing, route through the relink flow so we
    # verify the new folder and rewrite File.file_path records atomically.
    folder_path_in_payload = "folder_path" in update_fields
    new_folder_path = update_fields.get("folder_path")
    if (
        folder_path_in_payload
        and new_folder_path is not None
        and new_folder_path != current.folder_path
    ):
        relink_result = crud_deployment.relink_deployment(
            db, deployment_id, new_folder_path
        )
        if not relink_result.success:
            mismatches = (
                relink_result.verify_result.mismatches
                if relink_result.verify_result
                else []
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Folder verification failed",
                    "mismatches": mismatches,
                },
            )
        # Remove folder_path from the remaining update payload so the
        # second pass doesn't clobber the relink's bookkeeping.
        deployment = DeploymentUpdate(
            **{k: v for k, v in update_fields.items() if k != "folder_path"}
        )

    try:
        db_deployment = crud_deployment.update_deployment(db, deployment_id, deployment)
        if db_deployment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment with id '{deployment_id}' not found",
            )
        return DeploymentResponse.model_validate(db_deployment)
    except IntegrityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database constraint violation",
        ) from e


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_deployment(deployment_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a deployment.

    Returns 404 if deployment doesn't exist.
    Cascades deletion to all files and events.
    """
    deleted = crud_deployment.delete_deployment(db, deployment_id)
    if not deleted:
        logger.warning(f"Cannot delete deployment: {deployment_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )
    logger.info(f"Deleted deployment: {deployment_id} (cascaded to files and events)")


@router.get("/{deployment_id}/stats", response_model=DeploymentWithStats)
def get_deployment_stats(deployment_id: str, db: Session = Depends(get_db)) -> DeploymentWithStats:
    """
    Get deployment with statistics.

    Returns deployment info plus counts of files, events, and detections.
    Returns 404 if deployment doesn't exist.
    """
    db_deployment = crud_deployment.get_deployment(db, deployment_id)
    if db_deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )

    stats = crud_deployment.get_deployment_stats(db, deployment_id)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )

    # Combine deployment data with stats
    deployment_dict = DeploymentResponse.model_validate(db_deployment).model_dump()
    deployment_dict.update(stats)

    return DeploymentWithStats(**deployment_dict)


@router.post("/{deployment_id}/check-folder", response_model=DeploymentResponse)
def check_deployment_folder(
    deployment_id: str, db: Session = Depends(get_db)
) -> DeploymentResponse:
    """
    Re-stat a deployment's folder_path and update its folder_status.

    Called on demand (e.g. when the user opens the edit modal) so the
    status badge reflects the current filesystem state.
    """
    db_deployment = crud_deployment.check_deployment_folder(db, deployment_id)
    if db_deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )
    return DeploymentResponse.model_validate(db_deployment)


@router.post("/{deployment_id}/preview-folder", response_model=FolderPreviewResponse)
def preview_deployment_folder(
    deployment_id: str, db: Session = Depends(get_db)
) -> FolderPreviewResponse:
    """
    Preview a deployment folder before running analysis.

    Scans the folder to count images/videos and check for GPS coordinates.
    Does NOT create File records - that happens after MegaDetector runs.

    Returns 404 if deployment doesn't exist.
    Returns 400 if deployment has no folder_path set.
    Returns 400 if folder doesn't exist or isn't accessible.
    """
    # Get deployment
    db_deployment = crud_deployment.get_deployment(db, deployment_id)
    if db_deployment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment with id '{deployment_id}' not found",
        )

    # Check folder_path is set
    if not db_deployment.folder_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deployment has no folder_path set. Please set folder_path first.",
        )

    # Scan folder
    try:
        logger.info(f"Scanning folder for deployment {deployment_id}: {db_deployment.folder_path}")
        preview = scan_folder(db_deployment.folder_path)
        logger.info(
            f"Folder scan complete for {deployment_id}: "
            f"{preview['image_count']} images, {preview['video_count']} videos"
        )
    except FileNotFoundError as e:
        logger.error(
            f"Folder not found for deployment {deployment_id}: {db_deployment.folder_path}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Folder not found: {str(e)}",
        ) from e
    except PermissionError as e:
        logger.error(
            f"Permission denied for deployment {deployment_id}: {db_deployment.folder_path}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(
            f"Error scanning folder for deployment {deployment_id}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scanning folder: {str(e)}",
        ) from e

    # TODO: Site matching based on GPS (for later)
    suggested_site_id = None

    # Convert to response schema
    return FolderPreviewResponse(
        image_count=preview["image_count"],
        video_count=preview["video_count"],
        total_count=preview["total_count"],
        gps_location=GPSCoordinates(**preview["gps_location"]) if preview["gps_location"] else None,
        suggested_site_id=suggested_site_id,
        sample_files=preview["sample_files"],
        start_date=preview["start_date"],
        end_date=preview["end_date"],
        missing_datetime=preview["missing_datetime"],
        datetime_validation_log=preview["datetime_validation_log"],
    )
