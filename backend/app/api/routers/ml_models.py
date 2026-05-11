"""
ML Models API endpoints for status checking and preparation.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
"""

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.ml.batch_size import (
    CLASSIFICATION_DEFAULT_CPU,
    CLASSIFICATION_DEFAULT_GPU,
    DETECTION_DEFAULT_CPU,
    DETECTION_DEFAULT_GPU,
    EMBEDDING_DEFAULT_CPU,
    EMBEDDING_DEFAULT_GPU,
)
from app.ml.environment_manager import EnvironmentManager
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage

logger = get_logger(__name__)
router = APIRouter(prefix="/api/ml", tags=["ML Models"])

# Global instances (lazy initialization to avoid blocking on import)
manifest_manager = None
env_manager = None
model_storage = None


def _get_managers():
    """Get or initialize global manager instances."""
    global manifest_manager, env_manager, model_storage

    if manifest_manager is None:
        manifest_manager = ManifestManager()
    if env_manager is None:
        env_manager = EnvironmentManager()
    if model_storage is None:
        model_storage = ModelStorage()

    return manifest_manager, env_manager, model_storage


class ModelStatusResponse(BaseModel):
    """Response for model status check."""

    model_id: str
    friendly_name: str
    weights_ready: bool
    env_ready: bool
    weights_size_mb: float | None
    status: Literal["ready", "needs_weights", "needs_env", "needs_both"]


class ModelPrepareResponse(BaseModel):
    """Response for model preparation request."""

    model_id: str
    message: str
    task_id: str


class ModelInfo(BaseModel):
    """Model information for UI display."""

    model_id: str
    friendly_name: str
    emoji: str
    type: Literal["detection", "classification", "embedding"]
    description: str
    description_short: str | None = None
    developer: str | None = None
    owner: str | None = None
    info_url: str | None = None
    citation: str | None = None
    license: str | None = None
    min_app_version: str | None = None
    embedding_dim: int | None = None
    # Geographic region the cls model is trained for. Drives the
    # grouping in the classification dropdown. None for detection /
    # embedding models, and for any legacy cls manifest without a
    # region annotation (those fall into a synthetic "Other" group on
    # the frontend).
    region: str | None = None
    # Per-pipeline default batch sizes used when the project leaves the
    # batch_size override unset. Same value for every model in the same
    # pipeline today; comes from app.ml.batch_size constants.
    default_batch_size_gpu: int
    default_batch_size_cpu: int


# Lookup table mirroring the constants in app.ml.batch_size, used to
# populate ModelInfo.default_batch_size_* by pipeline type.
_DEFAULT_BATCH_SIZES_BY_TYPE: dict[str, tuple[int, int]] = {
    "detection": (DETECTION_DEFAULT_GPU, DETECTION_DEFAULT_CPU),
    "classification": (CLASSIFICATION_DEFAULT_GPU, CLASSIFICATION_DEFAULT_CPU),
    "embedding": (EMBEDDING_DEFAULT_GPU, EMBEDDING_DEFAULT_CPU),
}


@router.get("/models/{model_id}/status", response_model=ModelStatusResponse)
async def get_model_status(model_id: str) -> ModelStatusResponse:
    """
    Check if model weights and environment are ready.

    Returns status indicating what needs to be prepared.
    """
    try:
        # Initialize managers if needed
        manifest_mgr, env_mgr, storage = _get_managers()

        # Get model manifest
        manifest = manifest_mgr.get_model(model_id)

        # Check weights status
        weights_ready = storage.check_weights_ready(manifest)

        # Check environment status
        env_ready = False
        try:
            env_name = f"env-{manifest.env}"
            env_path = env_mgr.envs_dir / env_name
            if env_path.exists():
                env_ready = env_mgr._validate_env(env_path)
        except Exception as e:
            logger.warning(f"Failed to check environment status: {e}")
            env_ready = False

        # Get weights size if available
        weights_size = storage.get_weights_size(manifest)

        # Determine overall status
        if weights_ready and env_ready:
            overall_status = "ready"
        elif not weights_ready and not env_ready:
            overall_status = "needs_both"
        elif not weights_ready:
            overall_status = "needs_weights"
        else:
            overall_status = "needs_env"

        return ModelStatusResponse(
            model_id=model_id,
            friendly_name=manifest.friendly_name,
            weights_ready=weights_ready,
            env_ready=env_ready,
            weights_size_mb=weights_size,
            status=overall_status,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from None
    except Exception as e:
        logger.error(f"Failed to check model status for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check model status: {e}",
        ) from None


@router.post("/models/{model_id}/prepare", response_model=ModelPrepareResponse)
async def prepare_model(model_id: str) -> ModelPrepareResponse:
    """
    Prepare model by downloading weights and building environment.

    Sequential process:
    1. Download weights (if classification model)
    2. Build environment with micromamba

    Progress updates sent via WebSocket at /ws/ml/prepare/{model_id}
    """
    try:
        # Initialize managers if needed
        _get_managers()

        # Get model manifest
        manifest = manifest_manager.get_model(model_id)

        # Use model_id as task_id for WebSocket tracking
        task_id = model_id

        # Register worker to start when frontend sends "ready" over WebSocket
        ws_manager.register_start(
            task_id,
            lambda mid=model_id, m=manifest, tid=task_id: _prepare_model_task(mid, m, tid),
        )

        return ModelPrepareResponse(
            model_id=model_id,
            message="Model preparation started",
            task_id=task_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from None
    except Exception as e:
        logger.error(f"Failed to start preparation for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start preparation: {e}",
        ) from None


@router.post("/models/{model_id}/prepare-weights", response_model=ModelPrepareResponse)
async def prepare_model_weights(model_id: str) -> ModelPrepareResponse:
    """
    Download model weights only (without building environment).

    Progress updates sent via WebSocket at /ws/ml/prepare/{model_id}
    """
    try:
        # Initialize managers if needed
        _get_managers()

        # Get model manifest
        manifest = manifest_manager.get_model(model_id)

        # Use model_id as task_id for WebSocket tracking
        task_id = f"{model_id}-weights"

        # Register worker to start when frontend sends "ready" over WebSocket
        ws_manager.register_start(
            task_id,
            lambda mid=model_id, m=manifest, tid=task_id: _prepare_weights_task(mid, m, tid),
        )

        return ModelPrepareResponse(
            model_id=model_id,
            message="Weights download started",
            task_id=task_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from None
    except Exception as e:
        logger.error(f"Failed to start weights download for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start weights download: {e}",
        ) from None


@router.post("/models/{model_id}/redownload", status_code=status.HTTP_202_ACCEPTED)
async def redownload_model(model_id: str) -> dict[str, str]:
    """
    Force-redownload a model's files when the upstream HF revision has
    moved past the locally recorded SHA. Wipes everything in the model
    directory except `manifest.json` (so the catalog stub survives) and
    spawns the download as a fire-and-forget asyncio task. Returns 202
    immediately; the next `ModelCatalogUpdater.sync()` (i.e. the next
    app launch) will record the new SHA and clear the drift entry.

    No live progress is surfaced. The drift toast that triggered this
    is intentionally minimal; users who want to monitor a download
    closely should use the regular setup wizard / project flow that
    already wires WebSocket progress.
    """
    try:
        _get_managers()
        manifest = manifest_manager.get_model(model_id)

        # Run the wipe + download in a worker thread so the request
        # returns immediately. download_weights handles its own
        # cleanup on failure; failures here just mean drift stays
        # flagged on the next startup, which the user can retry.
        async def _run() -> None:
            try:
                await asyncio.to_thread(
                    model_storage.download_weights, manifest, None, True
                )
                logger.info(f"Force re-download of {model_id} complete")
            except Exception as e:
                logger.error(
                    f"Force re-download of {model_id} failed: {e}",
                    exc_info=True,
                )

        asyncio.create_task(_run())
        return {"status": "started", "model_id": model_id}

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from None
    except Exception as e:
        logger.error(f"Failed to start re-download for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start re-download: {e}",
        ) from None


@router.post("/models/{model_id}/prepare-env", response_model=ModelPrepareResponse)
async def prepare_model_environment(model_id: str) -> ModelPrepareResponse:
    """
    Build model environment only (without downloading weights).

    Progress updates sent via WebSocket at /ws/ml/prepare/{model_id}
    """
    try:
        # Initialize managers if needed
        _get_managers()

        # Get model manifest
        manifest = manifest_manager.get_model(model_id)

        # Use model_id as task_id for WebSocket tracking
        task_id = f"{model_id}-env"

        # Register worker to start when frontend sends "ready" over WebSocket
        ws_manager.register_start(
            task_id,
            lambda mid=model_id, m=manifest, tid=task_id: _prepare_env_task(mid, m, tid),
        )

        return ModelPrepareResponse(
            model_id=model_id,
            message="Environment build started",
            task_id=task_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from None
    except Exception as e:
        logger.error(f"Failed to start environment build for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start environment build: {e}",
        ) from None


async def _prepare_model_task(model_id: str, manifest, task_id: str) -> None:
    """
    Background task to prepare model (weights + environment).

    Args:
        model_id: Model ID
        manifest: Model manifest
        task_id: Task ID for WebSocket tracking
    """
    try:
        await ws_manager.send_progress(task_id, "Starting model preparation...", 0.0)

        # Get the current event loop for use in thread callbacks
        loop = asyncio.get_running_loop()

        # Check what needs to be prepared
        needs_weights = not model_storage.check_weights_ready(manifest)
        env_name = f"env-{manifest.env}"
        env_path = env_manager.envs_dir / env_name
        needs_env = not (env_path.exists() and env_manager._validate_env(env_path))

        # Dynamically allocate progress ranges based on what's needed
        if needs_weights and needs_env:
            # Both needed: weights 0-50%, env 50-100%
            weights_range = (0.0, 0.5)
            env_range = (0.5, 1.0)
        elif needs_weights:
            # Only weights: gets full 0-100%
            weights_range = (0.0, 1.0)
            env_range = None
        elif needs_env:
            # Only env: gets full 0-100%
            weights_range = None
            env_range = (0.0, 1.0)
        else:
            # Nothing needed (already prepared)
            await ws_manager.send_complete(
                task_id,
                success=True,
                message="Model already prepared",
                data={"model_id": model_id},
            )
            return

        # Step 1: Download weights from HuggingFace (if needed)
        if needs_weights:
            await ws_manager.send_progress(
                task_id, "Downloading model weights from HuggingFace...", weights_range[0] + 0.05
            )

            def weights_progress(message: str, progress: float):
                """Sync callback for weight download progress."""
                # Map to dynamic range
                start, end = weights_range
                mapped_progress = start + (progress * (end - start))

                # Add step prefix for clarity
                step_prefix = "Step 1/2 - " if needs_env else ""
                formatted_message = f"{step_prefix}{message}"

                # Schedule coroutine from thread using run_coroutine_threadsafe
                asyncio.run_coroutine_threadsafe(
                    ws_manager.send_progress(task_id, formatted_message, mapped_progress), loop
                )

            # Download weights (blocking call in thread pool)
            await asyncio.to_thread(model_storage.download_weights, manifest, weights_progress)

            await ws_manager.send_progress(task_id, "Weights downloaded", weights_range[1])

        # Step 2: Build environment (if needed)
        if needs_env:
            # Send initial progress for env build
            initial_msg = (
                "Step 2/2 - Building environment..." if needs_weights else "Building environment..."
            )
            await ws_manager.send_progress(task_id, initial_msg, env_range[0] + 0.01)

            def env_progress(message: str, progress: float):
                """Sync callback for environment build progress."""
                # Map to dynamic range
                start, end = env_range
                mapped_progress = start + (progress * (end - start))

                # Add step prefix (CSS will handle truncation in UI)
                if needs_weights:
                    prefix = "Step 2/2 - Installing: "
                else:
                    prefix = "Installing: "

                formatted_message = f"{prefix}{message}"
                logger.info(
                    f"Environment progress: {formatted_message} "
                    f"({progress:.1%} -> {mapped_progress:.1%})"
                )

                # Schedule coroutine from thread using run_coroutine_threadsafe
                asyncio.run_coroutine_threadsafe(
                    ws_manager.send_progress(task_id, formatted_message, mapped_progress), loop
                )

            # Build environment (blocking call in thread pool)
            await asyncio.to_thread(env_manager.get_or_create_env, manifest, env_progress)

        # No torch.hub pre-warm: DINOv2 architecture now ships inside each
        # Addax-Data-Science/DINOV2-* HF repo alongside the .pth weights,
        # and embedding_script.py loads it via source="local". So this step
        # no longer needs network access to github.com.

        await ws_manager.send_complete(
            task_id,
            success=True,
            message="Model preparation complete",
            data={"model_id": model_id},
        )

        logger.info(f"Model {model_id} prepared successfully")

    except Exception as e:
        logger.error(f"Failed to prepare model {model_id}: {e}", exc_info=True)
        await ws_manager.send_error(task_id, f"Preparation failed: {e}")


async def _prepare_weights_task(model_id: str, manifest, task_id: str) -> None:
    """
    Background task to download model weights only.

    Args:
        model_id: Model ID
        manifest: Model manifest
        task_id: Task ID for WebSocket tracking
    """
    try:
        await ws_manager.send_progress(task_id, "Starting weights download...", 0.0)

        # Get the current event loop for use in thread callbacks
        loop = asyncio.get_running_loop()

        def weights_progress(message: str, progress: float):
            """Sync callback for weight download progress."""
            # Schedule coroutine from thread using run_coroutine_threadsafe
            asyncio.run_coroutine_threadsafe(
                ws_manager.send_progress(task_id, message, progress), loop
            )

        # Download weights (blocking call in thread pool)
        await asyncio.to_thread(
            model_storage.download_weights, manifest, weights_progress
        )

        await ws_manager.send_complete(
            task_id,
            success=True,
            message="Weights download complete",
            data={"model_id": model_id},
        )

        logger.info(f"Weights for {model_id} downloaded successfully")

    except Exception as e:
        logger.error(f"Failed to download weights for {model_id}: {e}", exc_info=True)
        await ws_manager.send_error(task_id, f"Weights download failed: {e}")


async def _prepare_env_task(model_id: str, manifest, task_id: str) -> None:
    """
    Background task to build model environment only.

    Args:
        model_id: Model ID
        manifest: Model manifest
        task_id: Task ID for WebSocket tracking
    """
    try:
        await ws_manager.send_progress(task_id, "Starting environment build...", 0.0)

        # Get the current event loop for use in thread callbacks
        loop = asyncio.get_running_loop()

        def env_progress(message: str, progress: float):
            """Sync callback for environment build progress."""
            logger.info(f"Environment progress: {message} ({progress:.1%})")
            # Schedule coroutine from thread using run_coroutine_threadsafe
            asyncio.run_coroutine_threadsafe(
                ws_manager.send_progress(task_id, message, progress), loop
            )

        # Build environment (blocking call in thread pool)
        await asyncio.to_thread(env_manager.get_or_create_env, manifest, env_progress)

        await ws_manager.send_complete(
            task_id,
            success=True,
            message="Environment build complete",
            data={"model_id": model_id},
        )

        logger.info(f"Environment for {model_id} built successfully")

    except Exception as e:
        logger.error(f"Failed to build environment for {model_id}: {e}", exc_info=True)
        await ws_manager.send_error(task_id, f"Environment build failed: {e}")


@router.get("/updates")
def get_model_updates(request: Request) -> dict:
    """
    Get new models discovered during last startup check.

    Returns:
        Dict with new_models list and checked_at timestamp
    """
    # Access app.state from request
    updates = getattr(request.app.state, "model_updates", {"new_models": [], "checked_at": None})
    return updates


@router.get("/models/detection", response_model=list[ModelInfo])
def list_detection_models() -> list[ModelInfo]:
    """
    List all available detection models.

    Returns model metadata for UI dropdowns, sorted alphabetically by friendly_name.
    """
    manifest_mgr, _, _ = _get_managers()
    models = manifest_mgr.get_detection_models()

    det_gpu, det_cpu = _DEFAULT_BATCH_SIZES_BY_TYPE["detection"]
    model_list = [
        ModelInfo(
            model_id=manifest.model_id,
            friendly_name=manifest.friendly_name,
            emoji=manifest.emoji,
            type="detection",
            description=manifest.description or "",
            description_short=getattr(manifest, "description_short", None),
            developer=manifest.developer,
            owner=getattr(manifest, "owner", None),
            info_url=manifest.info_url,
            citation=getattr(manifest, "citation", None),
            license=getattr(manifest, "license", None),
            min_app_version=manifest.min_app_version,
            default_batch_size_gpu=det_gpu,
            default_batch_size_cpu=det_cpu,
        )
        for manifest in models.values()
    ]

    # Sort by user-friendly order: MD5A, MD5B first, then MD1000 models by accuracy (best to lowest)
    sort_order = {
        "MD5A-0-0": 0,
        "MD5B-0-0": 1,
        "MD1000-REDWOOD-0-0": 2,
        "MD1000-CEDAR-0-0": 3,
        "MD1000-LARCH-0-0": 4,
        "MD1000-SORREL-0-0": 5,
        "MD1000-SPRUCE-0-0": 6,
    }
    return sorted(model_list, key=lambda m: sort_order.get(m.model_id, 999))


@router.get("/models/classification", response_model=list[ModelInfo])
def list_classification_models() -> list[ModelInfo]:
    """
    List all available classification models.

    Returns model metadata for UI dropdowns, sorted alphabetically by friendly_name.
    Includes a "None" option for projects without classification.
    """
    manifest_mgr, _, _ = _get_managers()
    models = manifest_mgr.get_classification_models()

    cls_gpu, cls_cpu = _DEFAULT_BATCH_SIZES_BY_TYPE["classification"]

    # Add "None" option first
    result = [
        ModelInfo(
            model_id="none",
            friendly_name="No classification",
            emoji="⊘",
            type="classification",
            description="Run detection only, without species classification",
            default_batch_size_gpu=cls_gpu,
            default_batch_size_cpu=cls_cpu,
        )
    ]

    # Add actual classification models. Sorted by (region_order,
    # friendly_name) so the frontend can group by region while
    # keeping each group alphabetical. Region order: global first,
    # then continents alphabetical, then a synthetic "other" bucket
    # for legacy manifests that haven't been annotated yet.
    region_order = {
        "global": 0,
        "africa": 1,
        "americas": 2,
        "asia": 3,
        "europe": 4,
        "oceania": 5,
    }
    model_list = [
        ModelInfo(
            model_id=manifest.model_id,
            friendly_name=manifest.friendly_name,
            emoji=manifest.emoji,
            type="classification",
            description=manifest.description or "",
            description_short=getattr(manifest, "description_short", None),
            developer=manifest.developer,
            owner=getattr(manifest, "owner", None),
            info_url=manifest.info_url,
            citation=getattr(manifest, "citation", None),
            license=getattr(manifest, "license", None),
            min_app_version=manifest.min_app_version,
            region=getattr(manifest, "region", None),
            default_batch_size_gpu=cls_gpu,
            default_batch_size_cpu=cls_cpu,
        )
        for manifest in models.values()
    ]
    result.extend(
        sorted(
            model_list,
            key=lambda m: (region_order.get(m.region or "", 99), m.friendly_name),
        )
    )

    return result


@router.get("/models/embedding", response_model=list[ModelInfo])
def list_embedding_models() -> list[ModelInfo]:
    """
    List all available embedding models.

    Returns model metadata for UI dropdowns, sorted by embedding_dim (smallest first).
    Includes a "No embeddings" option as first item.
    """
    manifest_mgr, _, _ = _get_managers()
    models = manifest_mgr.get_embedding_models()

    emb_gpu, emb_cpu = _DEFAULT_BATCH_SIZES_BY_TYPE["embedding"]

    # Add "None" option first
    result = [
        ModelInfo(
            model_id="none",
            friendly_name="No embeddings",
            emoji="⊘",
            type="embedding",
            description="Skip embedding computation",
            default_batch_size_gpu=emb_gpu,
            default_batch_size_cpu=emb_cpu,
        )
    ]

    # Add actual embedding models, sorted by embedding_dim (smallest first)
    model_list = [
        ModelInfo(
            model_id=manifest.model_id,
            friendly_name=manifest.friendly_name,
            emoji=manifest.emoji,
            type="embedding",
            description=manifest.description or "",
            description_short=getattr(manifest, "description_short", None),
            developer=manifest.developer,
            owner=getattr(manifest, "owner", None),
            info_url=manifest.info_url,
            citation=getattr(manifest, "citation", None),
            license=getattr(manifest, "license", None),
            min_app_version=manifest.min_app_version,
            embedding_dim=manifest.embedding_dim,
            default_batch_size_gpu=emb_gpu,
            default_batch_size_cpu=emb_cpu,
        )
        for manifest in models.values()
    ]

    result.extend(sorted(model_list, key=lambda m: m.embedding_dim or 0))

    return result


@router.get("/models/{model_id}/taxonomy")
def get_model_taxonomy(model_id: str):
    """
    Get taxonomy tree for a classification model.

    Returns the hierarchical taxonomy structure and a flat list of all species.
    Reads from ~/AddaxAI/models/cls/{model_id}/taxonomy.csv

    Args:
        model_id: Classification model identifier (e.g., "NAM-ADS-v1")

    Returns:
        {
            "tree": list[TaxonomyNode],  # Hierarchical tree structure
            "all_classes": list[str]      # Flat list of all model_class values
        }

    Raises:
        404: If model not found or no taxonomy.csv exists
        500: If taxonomy.csv parsing fails
    """
    from app.core.config import get_settings
    from app.ml.taxonomy_parser import get_all_leaf_classes, parse_taxonomy_csv

    settings = get_settings()
    manifest_mgr, _, _ = _get_managers()

    # Validate model exists
    try:
        manifest = manifest_mgr.get_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None

    # Only classification models have taxonomy
    if manifest.model_category == "detection":
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_id} is a detection model and does not have taxonomy",
        )

    # Find taxonomy.csv in model directory
    # Look in ~/AddaxAI/models/cls/{model_id}/taxonomy.csv
    taxonomy_path = settings.user_data_dir / "models" / "cls" / model_id / "taxonomy.csv"

    if not taxonomy_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Taxonomy file not found for model {model_id}. "
            f"Expected at: {taxonomy_path}",
        )

    try:
        tree = parse_taxonomy_csv(taxonomy_path)
        all_classes = get_all_leaf_classes(tree)

        return {"tree": tree, "all_classes": all_classes}
    except Exception as e:
        logger.error(f"Failed to parse taxonomy for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse taxonomy: {str(e)}") from None


@router.get("/models/{model_id}/geofence")
def get_model_geofence(
    model_id: str,
    country: str | None = None,
    state: str | None = None,
):
    """
    Get geofence data for a classification model.

    If the model has a geofence file, returns available countries and
    optionally the allowed/excluded labels for a specific country.

    Args:
        model_id: Classification model ID
        country: Optional ISO country code to filter labels
        state: Optional US state code (only when country=USA)

    Returns:
        Without country param:
            {"has_geofence": true, "countries": {...}, "us_states": {...}}
        With country param:
            {"has_geofence": true, "allowed_labels": [...],
             "excluded_count": N, "total_count": N}
        If no geofence:
            {"has_geofence": false}
    """
    from app.core.config import get_settings
    from app.ml.geofence import (
        compute_excluded_classes,
        find_geofence_file,
        get_all_labels,
        get_allowed_labels,
    )

    settings = get_settings()
    model_dir = settings.user_data_dir / "models" / "cls" / model_id

    if not model_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model directory not found: {model_id}",
        )

    geofence_path = find_geofence_file(model_dir)
    if geofence_path is None:
        return {"has_geofence": False}

    if country is None:
        from app.ml.data.countries import countries_data, us_states_data

        return {
            "has_geofence": True,
            "countries": countries_data,
            "us_states": us_states_data,
        }

    try:
        allowed = get_allowed_labels(model_dir, country, state)
        all_labels = get_all_labels(model_dir)
        excluded = compute_excluded_classes(model_dir, country, state)

        return {
            "has_geofence": True,
            "allowed_labels": allowed,
            "excluded_labels": excluded,
            "excluded_count": len(excluded),
            "total_count": len(all_labels),
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail=str(e)
        ) from None
    except Exception as e:
        logger.error(f"Geofence computation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute geofence: {str(e)}",
        ) from None
