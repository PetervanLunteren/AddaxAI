"""
Project API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.crud import project as crud_project
from app.api.schemas.project import (
    CustomLabelCreate,
    CustomLabelResponse,
    CustomLabelUpdate,
    GBIFSuggestion,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    ProjectWithStats,
)
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.models import Deployment, Detection, File, Job, Site
from app.models.detection_embedding import DetectionEmbedding
from app.models.label_taxonomy import LabelTaxonomy

logger = get_logger(__name__)
router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("", response_model=list[ProjectResponse])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectResponse]:
    """
    List all projects.

    Returns empty list if no projects exist.
    """
    projects = crud_project.get_projects(db)
    return [ProjectResponse.model_validate(p) for p in projects]


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
    manifest_mgr = ManifestManager(settings.user_data_dir / "models")

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
    return ProjectResponse.model_validate(db_project)


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
        manifest_mgr = ManifestManager(settings.user_data_dir / "models")
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
                    settings.user_data_dir / "models" / "cls"
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

    try:
        db_project = crud_project.update_project(db, project_id, project)
        if db_project is None:
            logger.warning(f"Cannot update project: {project_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with id '{project_id}' not found",
            )
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
    # Collect deployment folder paths before cascade deletes them
    deployments = (
        db.query(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .all()
    )
    folder_paths = [Path(d.folder_path) for d in deployments]

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

    # Clean up project artifacts from each deployment folder
    for folder_path in folder_paths:
        project_artifacts = folder_path / ".addaxai" / "projects" / project_id
        if project_artifacts.exists():
            shutil.rmtree(project_artifacts)
            logger.info(f"Cleaned up artifacts: {project_artifacts}")

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


@router.get("/{project_id}/detection-stats")
def get_detection_stats(project_id: str, db: Session = Depends(get_db)) -> dict:
    """
    Get detection category statistics for a project.

    Returns counts by category (animal, person, vehicle).
    """
    # Query detection counts grouped by category
    stats = (
        db.query(Detection.category, func.count(Detection.id).label("count"))
        .join(File)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
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
    """
    count = (
        db.query(func.count(Detection.id))
        .join(File)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .filter(Detection.confidence >= threshold)
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
    """
    query = (
        db.query(Detection.label, func.count(Detection.id).label("count"))
        .join(File)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .filter(Detection.label.isnot(None))
    )
    if threshold > 0:
        query = query.filter(Detection.confidence >= threshold)
    stats = (
        query
        .group_by(Detection.label)
        .order_by(func.count(Detection.id).desc())
        .all()
    )

    return [{"label": label_name, "count": count} for label_name, count in stats]


@router.get("/{project_id}/independent-event-stats")
def get_independent_event_stats(
    project_id: str,
    interval: float = 1800,
    threshold: float = 0.0,
    db: Session = Depends(get_db),
) -> dict:
    """
    Count independent events per label for a project.

    Groups consecutive detections of the same label within a deployment
    that are within `interval` seconds of each other as a single event.
    Optionally filters by detection confidence threshold.

    Returns {total: int, labels: [{label, count}]}.
    """
    sql = text("""
        WITH ordered AS (
            SELECT
                d.label,
                dep.id AS deployment_id,
                f.timestamp,
                LAG(f.timestamp) OVER (
                    PARTITION BY dep.id, d.label
                    ORDER BY f.timestamp
                ) AS prev_timestamp
            FROM detections d
            JOIN files f ON d.file_id = f.id
            JOIN deployments dep ON f.deployment_id = dep.id
            JOIN sites s ON dep.site_id = s.id
            WHERE s.project_id = :project_id
              AND d.label IS NOT NULL
              AND (:threshold <= 0 OR d.confidence >= :threshold)
        ),
        events AS (
            SELECT label
            FROM ordered
            WHERE prev_timestamp IS NULL
               OR (julianday(timestamp) - julianday(prev_timestamp)) * 86400 > :interval
        )
        SELECT label, COUNT(*) AS event_count
        FROM events
        GROUP BY label
        ORDER BY event_count DESC
    """)

    rows = db.execute(
        sql,
        {"project_id": project_id, "interval": interval, "threshold": threshold},
    ).fetchall()

    total = sum(row[1] for row in rows)
    label_counts = [{"label": row[0], "count": row[1]} for row in rows]

    return {"total": total, "labels": label_counts}


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

    # Check if already exists (case-insensitive) in current model taxonomy
    # or among this project's custom labels
    existing = (
        db.query(LabelTaxonomy)
        .filter(
            func.lower(LabelTaxonomy.name) == name.lower(),
            (
                (LabelTaxonomy.classification_model_id == model_id)
                | (LabelTaxonomy.project_id == project_id)
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
                        .join(Site)
                        .filter(Site.project_id == project_id)
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

    # Ensure all detections with this label name point to this taxonomy row
    project_file_ids = (
        db.query(File.id)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
    )
    (
        db.query(Detection)
        .filter(
            Detection.label == row.name,
            Detection.label_taxonomy_id != label_id,
            Detection.file_id.in_(project_file_ids),
        )
        .update(
            {Detection.label_taxonomy_id: label_id},
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
    """Delete all embeddings for a project via Detection→File→Deployment→Site chain."""
    detection_ids = (
        db.query(Detection.id)
        .join(File)
        .join(Deployment)
        .join(Site)
        .filter(Site.project_id == project_id)
        .subquery()
    )
    count = (
        db.query(DetectionEmbedding)
        .filter(DetectionEmbedding.detection_id.in_(db.query(detection_ids.c.id)))
        .delete(synchronize_session=False)
    )
    db.commit()
    return count


@router.post(
    "/{project_id}/re-embed",
    status_code=status.HTTP_202_ACCEPTED,
)
async def re_embed_detections(
    project_id: str,
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
        },
    )
    job = crud_job.create_job(db, job_data)
    logger.info(f"Created re-embedding job {job.id} for project {project_id}")

    from app.workers.embedding_worker import process_re_embedding_job

    ws_manager.register_start(job.id, lambda jid=job.id: process_re_embedding_job(jid))

    return {"message": "Re-embedding started", "job_id": job.id}


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
        .join(Site)
        .filter(Site.project_id == project_id)
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
