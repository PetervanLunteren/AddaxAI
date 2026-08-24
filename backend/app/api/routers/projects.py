"""
Project API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import project as crud_project
from app.api.crud.deployment import _delete_deployment_artifacts
from app.api.schemas.project import (
    CustomLabelCreate,
    CustomLabelResponse,
    CustomLabelUpdate,
    GBIFSuggestion,
    MissingModel,
    ProjectCreate,
    ProjectDuplicate,
    ProjectModelReadiness,
    ProjectResponse,
    ProjectUpdate,
    ProjectWithStats,
)
from app.core.confidence import MD_OUTPUT_CONFIDENCE_THRESHOLD
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.ml.detection_visibility import on_visible_frame
from app.models import Deployment, Detection, Event, File, Job, Project
from app.models.detection_embedding import DetectionEmbedding
from app.models.event_observation import EventObservation
from app.models.label_taxonomy import LabelTaxonomy

# Query-only type. The DB column is one of the two real modes, but the
# list endpoint also accepts `all` to bypass filtering.
ListProjectsMode = Literal["folder_run", "research", "all"]

logger = get_logger(__name__)
router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("", response_model=list[ProjectWithStats])
def list_projects(
    mode: ListProjectsMode = "research",
    db: Session = Depends(get_db),
) -> list[ProjectWithStats]:
    """
    List projects with statistics, filtered by workflow mode.

    Defaults to `mode='research'` so the Research projects list
    excludes folder runs. Pass `?mode=folder_run` to get the recent
    folder runs strip, or `?mode=all` to include both.

    Each project includes counts for sites, deployments, files,
    detections, and trap nights, scoped to the selected mode.
    """
    effective_mode = None if mode == "all" else mode
    projects = crud_project.get_projects(db, mode=effective_mode)
    all_stats = crud_project.get_all_projects_stats(db, mode=effective_mode)

    result: list[ProjectWithStats] = []
    empty_stats = {
        "site_count": 0,
        "deployment_count": 0,
        "file_count": 0,
        "observation_count": 0,
        "trap_nights": 0,
    }
    for p in projects:
        project_dict = ProjectResponse.model_validate(p).model_dump()
        project_dict.update(all_stats.get(p.id, empty_stats))
        result.append(ProjectWithStats(**project_dict))

    return result


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    project: ProjectCreate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """
    Create a new project.

    Validates that selected models exist before creating project.
    Normalizes "none" to NULL for classification_model_id.

    Returns 400 if model IDs are invalid.
    Returns 409 if project name already exists.
    """
    from app.core.config import get_settings
    from app.ml.manifest_manager import ManifestManager

    # Validate models exist
    settings = get_settings()
    manifest_mgr = ManifestManager(settings.models_dir)

    # Validate detection model
    try:
        manifest_mgr.get_model(project.detection_model_id)
    except ValueError:
        logger.warning(f"Invalid detection model: {project.detection_model_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Detection model '{project.detection_model_id}' not found",
        ) from None

    # Normalize "none" to NULL and validate classification model
    if project.classification_model_id == "none":
        project.classification_model_id = None
    elif project.classification_model_id is not None:
        try:
            manifest_mgr.get_model(project.classification_model_id)
        except ValueError:
            logger.warning(f"Invalid classification model: {project.classification_model_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Classification model '{project.classification_model_id}' not found",
            ) from None

    # Normalize "none" to NULL and validate embedding model
    if project.embedding_model_id == "none":
        project.embedding_model_id = None
    elif project.embedding_model_id is not None:
        try:
            manifest_mgr.get_model(project.embedding_model_id)
        except ValueError:
            logger.warning(f"Invalid embedding model: {project.embedding_model_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Embedding model '{project.embedding_model_id}' not found",
            ) from None

    # Auto-compute excluded_classes from geofence when country_code is set
    if (
        project.country_code
        and project.country_code.upper() not in ("NONE", "")
        and project.classification_model_id
    ):
        try:
            from app.ml.geofence import compute_excluded_classes, find_geofence_file

            model_dir = (
                settings.models_dir / "cls"
                / project.classification_model_id
            )
            if model_dir.exists() and find_geofence_file(model_dir):
                excluded = compute_excluded_classes(
                    model_dir, project.country_code, project.state_code
                )
                project.excluded_classes = excluded
                logger.info(
                    f"Geofence: excluded {len(excluded)} labels "
                    f"for {project.country_code}"
                )
        except Exception as e:
            logger.warning(f"Geofence computation failed: {e}")

    try:
        db_project = crud_project.create_project(db, project)
        logger.info(f"Created project: {project.name} (ID: {db_project.id})")
        return ProjectResponse.model_validate(db_project)
    except IntegrityError as e:
        logger.warning(f"Failed to create project '{project.name}': duplicate name")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project with name '{project.name}' already exists",
        ) from e


@router.post(
    "/{project_id}/duplicate",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
def duplicate_project(
    project_id: str,
    params: ProjectDuplicate,
    db: Session = Depends(get_db),
) -> ProjectResponse:
    """Duplicate an existing project's structure into a new project.

    Copies the chosen visible fields plus, per the flags, the processing
    settings, the sites, and the source's deployments re-queued for
    reprocessing. Analyzed results are never copied across projects.

    Returns 404 if the source project is missing, 409 on a duplicate name.
    """
    if params.classification_model_id == "none":
        params.classification_model_id = None

    try:
        new_project = crud_project.duplicate_project(db, project_id, params)
    except IntegrityError as e:
        logger.warning(
            f"Failed to duplicate project: name '{params.name}' already exists"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project with name '{params.name}' already exists",
        ) from e

    if new_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source project not found",
        )

    logger.info(
        f"Duplicated project {project_id} -> {new_project.id} "
        f"({new_project.name})"
    )
    return ProjectResponse.model_validate(new_project)


@router.get("/gbif/suggest", response_model=list[GBIFSuggestion])
def gbif_suggest(q: str) -> list[GBIFSuggestion]:
    """
    Proxy GBIF species search.

    Tries VERNACULAR first, falls back to a general search (all fields)
    when the vernacular query returns no results. This handles cases like
    "king fisher" (two words) that GBIF's vernacular index doesn't match
    but the general index resolves fine.

    Deduplicates by canonicalName and returns up to 5 suggestions.
    """
    import httpx

    client = httpx.Client(timeout=5.0)

    skip_ranks = {"SUBSPECIES", "VARIETY", "FORM"}

    def _usable(r: dict) -> bool:
        """Check if a GBIF result is usable (has class, not a subspecies)."""
        return bool(
            r.get("canonicalName") and r.get("class")
            and r.get("rank", "") not in skip_ranks
        )

    try:
        # Try vernacular name search first
        resp = client.get(
            "https://api.gbif.org/v1/species/search",
            params={"q": q, "limit": 10, "qField": "VERNACULAR"},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        # Fall back to general search if no usable vernacular results
        if not any(_usable(r) for r in results):
            resp = client.get(
                "https://api.gbif.org/v1/species/search",
                params={"q": q, "limit": 10},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
    except httpx.HTTPError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="GBIF service unavailable",
        ) from err
    finally:
        client.close()

    # Boost exact canonical name matches to the top
    q_lower = q.strip().lower()
    results.sort(key=lambda r: r.get("canonicalName", "").lower() != q_lower)

    # Deduplicate by canonicalName.
    seen: set[str] = set()
    suggestions: list[GBIFSuggestion] = []
    for r in results:
        if not _usable(r):
            continue
        canonical = r["canonicalName"]
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        # GBIF returns species as a full binomial ("Urechis caupo").
        # Strip the genus prefix so we store just the epithet ("caupo"),
        # consistent with the CSV/JSON taxonomy convention.
        gbif_genus = r.get("genus") or ""
        gbif_species = r.get("species") or ""
        if gbif_genus and gbif_species.lower().startswith(gbif_genus.lower()):
            gbif_species = gbif_species[len(gbif_genus):].strip()

        suggestions.append(GBIFSuggestion(
            gbif_key=r.get("key", 0),
            scientific_name=r.get("scientificName", canonical),
            canonical_name=canonical,
            rank=r.get("rank", "UNKNOWN"),
            taxon_class=r.get("class"),
            taxon_order=r.get("order"),
            taxon_family=r.get("family"),
            taxon_genus=r.get("genus"),
            taxon_species=gbif_species or None,
        ))
        if len(suggestions) >= 5:
            break

    return suggestions


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    """
    Get project by ID.

    Returns 404 if project doesn't exist.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        logger.warning(f"Project not found: {project_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )
    logger.info(
        f"GET project {project_id}: "
        f"{len(db_project.excluded_classes or [])} excluded_classes"
    )
    return ProjectResponse.model_validate(db_project)


@router.get("/{project_id}/model-readiness", response_model=ProjectModelReadiness)
def get_project_model_readiness(
    project_id: str, db: Session = Depends(get_db)
) -> ProjectModelReadiness:
    """Report which of this project's configured models still need setup.

    Drives the project-open dialog and the pre-analysis safety check.
    Returns `ready=true` only when every configured model has its
    weights and a valid env on disk; otherwise lists each missing piece
    so the UI can offer "Set up" affordances per model.
    """
    from app.ml.environment_manager import EnvironmentManager
    from app.ml.manifest_manager import ManifestManager
    from app.ml.model_storage import ModelStorage

    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    manifest_mgr = ManifestManager()
    env_mgr = EnvironmentManager()
    storage = ModelStorage(manifest_mgr.models_dir)

    configured_ids = [
        db_project.detection_model_id,
        db_project.classification_model_id,
        db_project.embedding_model_id,
    ]
    missing: list[MissingModel] = []
    for model_id in configured_ids:
        if not model_id:
            continue
        try:
            manifest = manifest_mgr.get_model(model_id)
        except Exception as e:
            # An unknown model id on the project (e.g. catalog removed it
            # between sessions) shows up as missing too, so the user
            # cannot silently start a job against a model that no longer
            # exists. Logged for diagnosability.
            logger.warning(
                f"Project {project_id} references unknown model {model_id}: {e}"
            )
            missing.append(
                MissingModel(
                    model_id=model_id,
                    friendly_name=model_id,
                    emoji="❓",
                    category="unknown",
                    needs_weights=True,
                    needs_env=True,
                )
            )
            continue

        needs_weights = not storage.check_weights_ready(manifest)
        env_path = env_mgr.envs_dir / f"env-{manifest.env}"
        needs_env = not (env_path.exists() and env_mgr._validate_env(env_path))
        if needs_weights or needs_env:
            missing.append(
                MissingModel(
                    model_id=manifest.model_id,
                    friendly_name=manifest.friendly_name,
                    emoji=manifest.emoji or "📦",
                    category=manifest.model_category,
                    needs_weights=needs_weights,
                    needs_env=needs_env,
                )
            )

    return ProjectModelReadiness(ready=len(missing) == 0, missing=missing)


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str, project: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectResponse:
    """
    Update an existing project.

    Returns 400 if all species are excluded.
    Returns 404 if project doesn't exist.
    Returns 409 if new name conflicts with existing project.
    """
    # Normalize "none" to NULL and validate embedding model
    if project.embedding_model_id == "none":
        project.embedding_model_id = None
    elif project.embedding_model_id is not None:
        from app.core.config import get_settings
        from app.ml.manifest_manager import ManifestManager

        settings = get_settings()
        manifest_mgr = ManifestManager(settings.models_dir)
        try:
            manifest_mgr.get_model(project.embedding_model_id)
        except ValueError:
            logger.warning(f"Invalid embedding model: {project.embedding_model_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Embedding model '{project.embedding_model_id}' not found",
            ) from None

    # Validate that not all species are excluded
    if project.excluded_classes is not None and len(project.excluded_classes) > 0:
        db_existing = crud_project.get_project(db, project_id)
        if db_existing and db_existing.classification_model_id:
            try:
                from app.core.config import get_settings
                from app.ml.taxonomy_parser import get_all_leaf_classes, parse_taxonomy_csv

                settings = get_settings()
                taxonomy_path = (
                    settings.models_dir / "cls"
                    / db_existing.classification_model_id / "taxonomy.csv"
                )
                if taxonomy_path.exists():
                    tree = parse_taxonomy_csv(taxonomy_path)
                    all_classes = get_all_leaf_classes(tree)
                    if len(project.excluded_classes) >= len(all_classes):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Cannot exclude all species. At least one must remain included.",
                        )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Could not validate excluded_classes: {e}")

    # Check if counting_threshold is being changed (affects MaxN)
    threshold_changing = (
        project.counting_threshold is not None
        and project.model_dump(exclude_unset=True).get("counting_threshold") is not None
    )

    # Debug: log what the frontend sent for excluded_classes
    update_fields = project.model_dump(exclude_unset=True)
    if "excluded_classes" in update_fields:
        exc = update_fields["excluded_classes"]
        logger.info(
            f"PATCH project {project_id}: excluded_classes has "
            f"{len(exc)} entries (first 5: {exc[:5]})"
        )
    else:
        logger.info(
            f"PATCH project {project_id}: excluded_classes NOT in payload"
        )

    try:
        db_project = crud_project.update_project(db, project_id, project)
        if db_project is None:
            logger.warning(f"Cannot update project: {project_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with id '{project_id}' not found",
            )

        # Debug: log what was stored
        logger.info(
            f"PATCH project {project_id}: DB now has "
            f"{len(db_project.excluded_classes or [])} excluded_classes"
        )

        # Recalculate MaxN if threshold changed
        if threshold_changing:
            from app.api.crud import file as crud_file
            from app.api.crud.event_observation import recalculate_max_n_for_project

            recalculate_max_n_for_project(db, project_id)
            # observation_type is threshold-aware (a file with only
            # sub-threshold boxes is "blank"), so a threshold change can
            # flip files between blank and animal / human / vehicle.
            crud_file.recalculate_observation_types_for_project(
                db, project_id
            )
            db.commit()

        logger.info(f"Updated project: {project_id}")
        return ProjectResponse.model_validate(db_project)
    except IntegrityError as e:
        logger.warning(f"Failed to update project {project_id}: duplicate name")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project name already exists",
        ) from e


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> None:
    """
    Delete a project.

    Returns 404 if project doesn't exist.
    Cascades deletion to all sites, deployments, files, etc.
    Also cleans up project-scoped artifacts from deployment folders.
    """
    # Collect deployment folder paths before cascade deletes them.
    # Only the column, never the entity: loading the rows would put every
    # deployment in the session, and `db.delete(project)` then cascades to
    # them in Python, one DELETE each, instead of leaving it to SQLite.
    # folder_path is nullable (a deployment can exist before its folder is
    # linked), and Path(None) raises. Same guard as crud/deployment.py.
    folder_paths = [
        Path(path)
        for (path,) in db.query(Deployment.folder_path)
        .filter(Deployment.project_id == project_id)
        .all()
        if path
    ]

    # Delete project from DB (cascades to sites, deployments, files, detections)
    deleted = crud_project.delete_project(db, project_id)
    if not deleted:
        logger.warning(f"Cannot delete project: {project_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    # Delete custom label taxonomy entries scoped to this project
    taxonomy_count = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.project_id == project_id)
        .delete(synchronize_session=False)
    )
    if taxonomy_count:
        db.commit()
        logger.info(f"Deleted {taxonomy_count} custom labels for project {project_id}")

    # Delete jobs associated with this project (after cascade removes detections
    # that have a FK to jobs)
    job_count = (
        db.query(Job)
        .filter(text("json_extract(payload, '$.project_id') = :pid"))
        .params(pid=project_id)
        .delete(synchronize_session=False)
    )
    if job_count:
        db.commit()
        logger.info(f"Deleted {job_count} jobs for project {project_id}")

    # Clean up project artifacts from each deployment folder.
    #
    # Through the shared helper, which swallows OS errors, because the rows
    # are already committed by now: a folder we cannot remove must not turn
    # a delete that succeeded into a 500 that says it failed. This used to
    # be an inline `shutil.rmtree`, and a `.addaxai` folder on a
    # disconnected external drive (the normal place for camera trap files)
    # returned "Internal Server Error" for a project that was already gone,
    # and skipped the cleanup for every remaining deployment as well.
    for folder_path in folder_paths:
        _delete_deployment_artifacts(str(folder_path), project_id)

    # Clean up thumbnail files. Best-effort for the same reason.
    from app.core.config import get_settings

    settings = get_settings()
    for subdir in ("project-images", "thumbnails"):
        thumb = settings.user_data_dir / subdir / f"{project_id}.jpg"
        try:
            if thumb.exists():
                thumb.unlink()
                logger.info(f"Deleted thumbnail: {thumb}")
        except OSError as e:
            logger.warning(f"Could not delete thumbnail {thumb}: {e}")

    logger.info(f"Deleted project: {project_id} (cascaded to all related data)")


@router.get("/{project_id}/stats", response_model=ProjectWithStats)
def get_project_stats(
    project_id: str, db: Session = Depends(get_db)
) -> ProjectWithStats:
    """
    Get project with statistics.

    Returns project info plus counts of sites, deployments, files, and detections.
    Returns 404 if project doesn't exist.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    stats = crud_project.get_project_stats(db, project_id)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    # Combine project data with stats
    project_dict = ProjectResponse.model_validate(db_project).model_dump()
    project_dict.update(stats)

    return ProjectWithStats(**project_dict)


@router.get("/{project_id}/thumbnail")
def get_project_thumbnail(
    project_id: str, db: Session = Depends(get_db)
) -> FileResponse:
    """Serve the project card thumbnail image."""
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    if not db_project.thumbnail_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No thumbnail set for this project",
        )

    thumb = Path(db_project.thumbnail_path)
    if not thumb.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thumbnail file missing from disk",
        )

    return FileResponse(
        path=str(thumb),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


_MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@router.post("/{project_id}/thumbnail")
def upload_project_thumbnail(
    project_id: str,
    file: UploadFile,
    db: Session = Depends(get_db),
) -> dict:
    """Upload a project card thumbnail image.

    Accepts JPEG or PNG, max 5 MB. Resizes to 512px wide and saves
    as JPEG.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are accepted",
        )

    contents = file.file.read()
    if len(contents) > _MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image must be smaller than 5 MB",
        )

    from app.core.config import get_settings
    from app.services.thumbnail_service import generate_thumbnail

    settings = get_settings()
    upload_dir = settings.user_data_dir / "project-images"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file to a temp location, then generate thumbnail
    raw_path = upload_dir / f"{project_id}_raw.tmp"
    raw_path.write_bytes(contents)

    try:
        dest = upload_dir / f"{project_id}.jpg"
        generate_thumbnail(raw_path, dest)
    finally:
        raw_path.unlink(missing_ok=True)

    # Remove any old auto-generated thumbnail
    auto_thumb = settings.user_data_dir / "thumbnails" / f"{project_id}.jpg"
    if auto_thumb.exists():
        auto_thumb.unlink()

    db_project.thumbnail_path = str(dest)
    db_project.updated_at_utc = datetime.now(UTC)
    db.commit()

    logger.info(f"Uploaded thumbnail for project {project_id}")
    return {"message": "Thumbnail uploaded"}


@router.delete(
    "/{project_id}/thumbnail",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_project_thumbnail(
    project_id: str, db: Session = Depends(get_db)
) -> None:
    """Remove the project card thumbnail."""
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    if db_project.thumbnail_path:
        thumb = Path(db_project.thumbnail_path)
        if thumb.exists():
            thumb.unlink()
        db_project.thumbnail_path = None
        db_project.updated_at_utc = datetime.now(UTC)
        db.commit()
        logger.info(f"Deleted thumbnail for project {project_id}")


@router.get("/{project_id}/detection-stats")
def get_detection_stats(project_id: str, db: Session = Depends(get_db)) -> dict:
    """
    Get detection category statistics for a project.

    Returns counts by category (animal, person, vehicle).
    Respects project detection threshold; verified detections always included.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    threshold = project.counting_threshold if project else 0.0

    stats = (
        db.query(Detection.category, func.count(Detection.id).label("count"))
        .join(File)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .filter(or_(Detection.confidence >= threshold, Detection.verified == True))  # noqa: E712
        .group_by(Detection.category)
        .all()
    )

    # Convert to dict
    result = {category: count for category, count in stats}

    return result


@router.get("/{project_id}/detection-count")
def get_detection_count(
    project_id: str,
    threshold: float = 0.0,
    db: Session = Depends(get_db),
) -> dict:
    """
    Get count of detections at or above a confidence threshold.

    Used by the frontend to show the impact of threshold changes.

    Counts only detections the user can actually reach, so this agrees
    with the Labels grid, the label tree and the exports. Without the
    frame gate a video project reported every sampled frame: 220 where
    the grid held 32, and the reprocess summary built on it promised
    changes to boxes nobody can open.
    """
    count = (
        db.query(func.count(Detection.id))
        .join(File)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .filter(or_(Detection.confidence >= threshold, Detection.verified == True))  # noqa: E712
        .filter(on_visible_frame())
        .scalar()
    ) or 0
    return {"count": count}


@router.get("/{project_id}/label-stats")
def get_label_stats(
    project_id: str,
    threshold: float = 0.0,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Get top label counts for a project.

    Returns list of {label, count} sorted by count descending.
    Only includes detections with a label classification.
    Optionally filters by confidence threshold.

    Frame-gated like every other user-facing count (see
    ``get_detection_count``): these numbers drive the "Effect on
    statistics" summary, so they have to describe labels the user can
    open and correct.
    """
    query = (
        db.query(Detection.label, func.count(Detection.id).label("count"))
        .join(File)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .filter(Detection.label.isnot(None))
        .filter(on_visible_frame())
    )
    if threshold > 0:
        query = query.filter(
            or_(Detection.confidence >= threshold, Detection.verified == True)  # noqa: E712
        )
    stats = (
        query
        .group_by(Detection.label)
        .order_by(func.count(Detection.id).desc())
        .all()
    )

    return [{"label": label_name, "count": count} for label_name, count in stats]


@router.get("/{project_id}/independent-observation-stats")
def get_independent_observation_stats(
    project_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Sum the effective per-event count across events, per label.

    Reads the materialized ``event_observations`` (the same source the
    dashboard "Observations" and exports use), so it honours human counts
    (``effective_count`` = ``human_count`` if set, else the AI ``max_n``)
    and human-added/removed species. Interval and threshold are baked
    into that materialized state at analysis/reprocess time, so they are
    not query parameters here. Animal labels only (person/vehicle are not
    part of label refinement). Returns ``{total, labels}``.
    """
    rows = (
        db.query(
            EventObservation.label,
            func.sum(EventObservation.effective_count).label("count"),
        )
        .join(Event, Event.id == EventObservation.event_id)
        .join(Deployment, Event.deployment_id == Deployment.id)
        .filter(Deployment.project_id == project_id)
        .filter(EventObservation.category == "animal")
        .filter(EventObservation.label.isnot(None))
        .group_by(EventObservation.label)
        .order_by(func.sum(EventObservation.effective_count).desc())
        .all()
    )

    total = sum(row[1] for row in rows)
    label_counts = [{"label": row[0], "count": int(row[1])} for row in rows]
    return {"total": total, "labels": label_counts}


@router.get("/{project_id}/regroup-preview")
def get_regroup_preview(
    project_id: str,
    independence_interval: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    How much count verification a change to `independence_interval` would
    reset. Changing the interval re-clusters events; an event's confirmation
    and manual counts only survive when its file set still forms one cluster.
    Returns ``{confirmed_at_risk, counts_at_risk, total_confirmed}`` so the UI
    can warn before regrouping. Read-only.
    """
    from app.api.crud import event as crud_event

    return crud_event.count_regroup_impact(db, project_id, independence_interval)


@router.post(
    "/{project_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_classifications(
    project_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Reprocess classifications for a project.

    Launches an async task that sends WebSocket progress updates.
    Returns immediately with a job_id for progress tracking.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    from app.api.crud import job as crud_job
    from app.api.schemas.job import JobCreate

    job_data = JobCreate(
        type="postprocessing",
        payload={"project_id": project_id},
    )
    job = crud_job.create_job(db, job_data)
    logger.info(f"Created postprocessing job {job.id} for project {project_id}")

    from app.workers.postprocessing_worker import process_postprocessing_job

    ws_manager.register_start(job.id, lambda jid=job.id: process_postprocessing_job(jid))

    return {"message": "Postprocessing started", "job_id": job.id}


@router.get("/{project_id}/label-taxonomy-map")
def get_label_taxonomy_map(
    project_id: str, db: Session = Depends(get_db)
) -> dict[str, dict[str, str | None]]:
    """
    Return taxonomy fields for every label in a project.

    Merges model-level taxonomy entries with project-scoped custom
    labels. The result is keyed by label name, with each value
    containing the five taxonomy ranks (nullable).
    """
    project = crud_project.get_project(db, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    model_id = project.classification_model_id

    rows = (
        db.query(LabelTaxonomy)
        .filter(
            (LabelTaxonomy.classification_model_id == model_id)
            | (
                (LabelTaxonomy.project_id == project_id)
                & (LabelTaxonomy.is_custom == True)  # noqa: E712
            )
        )
        .all()
    )

    result: dict[str, dict[str, str | None]] = {}
    for row in rows:
        result[row.name] = {
            "taxon_class": row.taxon_class,
            "taxon_order": row.taxon_order,
            "taxon_family": row.taxon_family,
            "taxon_genus": row.taxon_genus,
            "taxon_species": row.taxon_species,
            "taxon_variant": row.taxon_variant,
            "common_name": row.common_name,
            "scientific_name": row.scientific_name,
        }
    return result


@router.get("/{project_id}/custom-labels", response_model=list[CustomLabelResponse])
def list_custom_labels(
    project_id: str, db: Session = Depends(get_db)
) -> list[CustomLabelResponse]:
    """
    List custom labels for a project.

    Returns all user-defined custom label entries for this project.
    """
    rows = (
        db.query(LabelTaxonomy)
        .filter(
            LabelTaxonomy.project_id == project_id,
            LabelTaxonomy.is_custom == True,  # noqa: E712
        )
        .all()
    )
    return [CustomLabelResponse.model_validate(r) for r in rows]


@router.post(
    "/{project_id}/custom-labels",
    response_model=CustomLabelResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_custom_label(
    project_id: str,
    body: CustomLabelCreate,
    db: Session = Depends(get_db),
) -> CustomLabelResponse:
    """
    Add a custom label to a project.

    If the label name already exists (case-insensitive) in the model taxonomy
    or among this project's custom labels, returns the existing entry.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    name = body.name.strip()
    model_id = db_project.classification_model_id

    from app.ml.taxonomic_rollup import format_common_name
    from app.ml.taxonomy_db import BUILTIN_MODEL_ID

    # Check if already exists (case-insensitive) in the current model
    # taxonomy, among this project's custom labels, or among the builtin
    # rows (animal / person / vehicle).
    #
    # The builtin arm is not decoration. Without it, "vehicle" here made a
    # third row of that name beside the builtin one and the model's own,
    # all rank-less and all displaying "Vehicle". The label filter shows
    # one leaf per taxonomy row, so those render as identical entries the
    # user cannot tell apart, and relabelling resolves to whichever the
    # priority order picks. One name, one row.
    existing = (
        db.query(LabelTaxonomy)
        .filter(
            func.lower(LabelTaxonomy.name) == name.lower(),
            (
                (LabelTaxonomy.classification_model_id == model_id)
                | (LabelTaxonomy.project_id == project_id)
                | (LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID)
            ),
        )
        .first()
    )

    if existing:
        return CustomLabelResponse.model_validate(existing)

    new_label = LabelTaxonomy(
        is_custom=True,
        project_id=project_id,
        level="unknown",
        name=name,
        classification_model_id="",
        common_name=format_common_name(name),
        scientific_name=name.capitalize(),
    )
    db.add(new_label)
    db.commit()
    db.refresh(new_label)

    logger.info(f"Created custom label '{name}' for project {project_id}")
    return CustomLabelResponse.model_validate(new_label)


def _derive_taxonomy_level(body: CustomLabelUpdate) -> str:
    """Derive the most specific taxonomy level from populated fields."""
    if body.taxon_species:
        return "species"
    if body.taxon_genus:
        return "genus"
    if body.taxon_family:
        return "family"
    if body.taxon_order:
        return "order"
    if body.taxon_class:
        return "class"
    return "unknown"


@router.patch(
    "/{project_id}/custom-labels/{label_id}",
    response_model=CustomLabelResponse,
)
def update_custom_label(
    project_id: str,
    label_id: str,
    body: CustomLabelUpdate,
    db: Session = Depends(get_db),
) -> CustomLabelResponse:
    """
    Update taxonomy fields on a custom label.

    Derives the taxonomy level from the most specific populated field.
    """
    row = (
        db.query(LabelTaxonomy)
        .filter(
            LabelTaxonomy.id == label_id,
            LabelTaxonomy.project_id == project_id,
            LabelTaxonomy.is_custom == True,  # noqa: E712
        )
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom label not found",
        )

    # Handle name update with collision check
    if body.name is not None:
        new_name = body.name.strip()
        old_name = row.name
        if new_name.lower() != old_name.lower():
            collision = (
                db.query(LabelTaxonomy)
                .filter(
                    func.lower(LabelTaxonomy.name) == new_name.lower(),
                    LabelTaxonomy.id != label_id,
                    LabelTaxonomy.project_id == project_id,
                )
                .first()
            )
            if collision:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Label '{new_name}' already exists",
                )
        if new_name != old_name:
            # Update all detections in this project that reference the old name
            (
                db.query(Detection)
                .filter(
                    Detection.label == old_name,
                    Detection.file_id.in_(
                        db.query(File.id)
                        .join(Deployment)
                        .filter(Deployment.project_id == project_id)
                    ),
                )
                .update(
                    {Detection.label: new_name, Detection.label_taxonomy_id: label_id},
                    synchronize_session=False,
                )
            )
            logger.info(f"Renamed detections '{old_name}' -> '{new_name}' in project {project_id}")
        row.name = new_name

    row.taxon_class = body.taxon_class
    row.taxon_order = body.taxon_order
    row.taxon_family = body.taxon_family
    row.taxon_genus = body.taxon_genus
    row.taxon_species = body.taxon_species
    row.level = _derive_taxonomy_level(body)

    # Recompute both names from updated taxonomy fields
    from app.ml.taxonomic_rollup import (
        format_common_name,
        format_scientific_name_from_taxonomy_row,
    )

    row.common_name = format_common_name(row.name)
    row.scientific_name = format_scientific_name_from_taxonomy_row(
        row.name,
        body.taxon_genus,
        body.taxon_species,
        body.taxon_family,
        body.taxon_order,
        body.taxon_class,
    )

    # Ensure all detections with this label name point to this taxonomy
    # row and carry the updated names
    project_file_ids = (
        db.query(File.id)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
    )
    (
        db.query(Detection)
        .filter(
            Detection.label == row.name,
            Detection.file_id.in_(project_file_ids),
        )
        .update(
            {
                Detection.label_taxonomy_id: label_id,
                Detection.scientific_name: row.scientific_name,
                Detection.common_name: row.common_name,
            },
            synchronize_session=False,
        )
    )

    db.commit()
    db.refresh(row)

    logger.info(f"Updated custom label '{row.name}' -> level={row.level}")
    return CustomLabelResponse.model_validate(row)


@router.delete(
    "/{project_id}/custom-labels/{label_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_custom_label(
    project_id: str,
    label_id: str,
    db: Session = Depends(get_db),
) -> None:
    """Delete a custom label from a project."""
    row = (
        db.query(LabelTaxonomy)
        .filter(
            LabelTaxonomy.id == label_id,
            LabelTaxonomy.project_id == project_id,
            LabelTaxonomy.is_custom == True,  # noqa: E712
        )
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom label not found",
        )

    name = row.name

    # SET NULL the FK on detections that reference this taxonomy row
    (
        db.query(Detection)
        .filter(Detection.label_taxonomy_id == label_id)
        .update({Detection.label_taxonomy_id: None}, synchronize_session=False)
    )

    db.delete(row)
    db.commit()
    logger.info(f"Deleted custom label '{name}' from project {project_id}")


def _delete_project_embeddings(db: Session, project_id: str) -> int:
    """Delete all embeddings for a project via Detection→File→Deployment chain."""
    detection_ids = (
        db.query(Detection.id)
        .join(File)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .subquery()
    )
    count = (
        db.query(DetectionEmbedding)
        .filter(DetectionEmbedding.detection_id.in_(db.query(detection_ids.c.id)))
        .delete(synchronize_session=False)
    )
    db.commit()
    return count


class ReEmbedRequest(BaseModel):
    """Optional body for POST /{project_id}/re-embed.

    ``min_confidence`` overrides the project's classification gate as
    the embedding floor for this one job. The labels grid's
    "unprocessed detections" banner uses it to backfill embeddings for
    a below-gate confidence range the user chose to review; detections
    already embedded are skipped as usual, so the job only adds the
    missing tail.
    """

    min_confidence: float | None = Field(
        None, ge=MD_OUTPUT_CONFIDENCE_THRESHOLD, le=1.0
    )


@router.post(
    "/{project_id}/re-embed",
    status_code=status.HTTP_202_ACCEPTED,
)
async def re_embed_detections(
    project_id: str,
    payload: ReEmbedRequest | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """
    Re-embed all detections with the project's current embedding model.

    If embedding_model_id is None, deletes all embeddings inline.
    Otherwise, launches an async re-embedding job.
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    embedding_model_id = db_project.embedding_model_id

    # No embedding model → delete all embeddings inline
    if not embedding_model_id:
        count = _delete_project_embeddings(db, project_id)
        logger.info(f"Deleted {count} embeddings for project {project_id}")
        return {"message": f"Deleted {count} embeddings", "job_id": None}

    # Create re-embedding job
    from app.api.crud import job as crud_job
    from app.api.schemas.job import JobCreate

    job_data = JobCreate(
        type="re_embedding",
        payload={
            "project_id": project_id,
            "embedding_model_id": embedding_model_id,
            "min_confidence": payload.min_confidence if payload else None,
        },
    )
    job = crud_job.create_job(db, job_data)
    logger.info(f"Created re-embedding job {job.id} for project {project_id}")

    from app.workers.embedding_worker import process_re_embedding_job

    ws_manager.register_start(job.id, lambda jid=job.id: process_re_embedding_job(jid))

    return {"message": "Re-embedding started", "job_id": job.id}


@router.get("/{project_id}/deployments-without-site")
def get_deployments_without_site(
    project_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Report deployments in this project that have no camera site.

    Used by the pages that need camera lat/lon (Map, Activity overlap
    sun-time mode, Dashboard activity sun bands, CamtrapDP / GeoJSON
    exports) to render a banner pointing users at the deployments that
    will be silently excluded.

    Returns ``{"count": N, "deployment_ids": [...]}``. The list is
    small in practice (a handful of backlog folders, typically) and is
    used by the banner button to deep-link into the deployments page
    with the (no site) filter applied.
    """
    rows = (
        db.query(Deployment.id)
        .filter(Deployment.project_id == project_id)
        .filter(Deployment.site_id.is_(None))
        .all()
    )
    ids = [r[0] for r in rows]
    return {"count": len(ids), "deployment_ids": ids}


@router.get("/{project_id}/files-without-date")
def get_files_without_date(
    project_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Count files in this project with no capture date.

    Used by the CamtrapDP export dialog: the schema requires a timestamp
    on every media and observation record, so these files are left out
    of that export and the dialog warns about them up front.

    Returns ``{"count": N}``.
    """
    count = (
        db.query(func.count(File.id))
        .join(Deployment, File.deployment_id == Deployment.id)
        .filter(Deployment.project_id == project_id)
        .filter(File.captured_at_local.is_(None))
        .scalar()
    )
    return {"count": count or 0}


@router.get("/{project_id}/postprocessing-status")
def get_postprocessing_status(
    project_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Check whether postprocessing needs to be re-run.

    Compares current smoothing settings hash with stored hash.

    Returns:
        {needs_reprocessing: bool, has_classifications: bool}
    """
    db_project = crud_project.get_project(db, project_id)
    if db_project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id '{project_id}' not found",
        )

    # Check if any classifications exist for this project
    has_cls = (
        db.query(Detection.id)
        .join(File)
        .join(Deployment)
        .filter(Deployment.project_id == project_id)
        .filter(Detection.label.isnot(None))
        .limit(1)
        .first()
    ) is not None

    # Compute current hash and compare
    from app.ml.postprocessing import compute_postprocessing_settings_hash

    current_hash = compute_postprocessing_settings_hash(db_project)
    stored_hash = db_project.postprocessing_settings_hash

    needs_reprocessing = has_cls and (current_hash != stored_hash)

    return {
        "needs_reprocessing": needs_reprocessing,
        "has_classifications": has_cls,
    }
