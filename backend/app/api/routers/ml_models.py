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
        )
    except Exception as e:
        logger.error(f"Failed to check model status for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check model status: {e}",
        )


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

        # Start preparation in background
        asyncio.create_task(_prepare_model_task(model_id, manifest, task_id))

        return ModelPrepareResponse(
            model_id=model_id,
            message="Model preparation started",
            task_id=task_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to start preparation for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start preparation: {e}",
        )


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

        # Start weights download in background
        asyncio.create_task(_prepare_weights_task(model_id, manifest, task_id))

        return ModelPrepareResponse(
            model_id=model_id,
            message="Weights download started",
            task_id=task_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to start weights download for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start weights download: {e}",
        )


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

        # Start environment build in background
        asyncio.create_task(_prepare_env_task(model_id, manifest, task_id))

        return ModelPrepareResponse(
            model_id=model_id,
            message="Environment build started",
            task_id=task_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to start environment build for {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start environment build: {e}",
        )


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
            await asyncio.to_thread(
                model_storage.download_weights, manifest, weights_progress
            )

            await ws_manager.send_progress(task_id, "Weights downloaded", weights_range[1])

        # Step 2: Build environment (if needed)
        if needs_env:
            # Send initial progress for env build
            initial_msg = "Step 2/2 - Building environment..." if needs_weights else "Building environment..."
            await ws_manager.send_progress(
                task_id, initial_msg, env_range[0] + 0.01
            )

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
                logger.info(f"Environment progress: {formatted_message} ({progress:.1%} -> {mapped_progress:.1%})")

                # Schedule coroutine from thread using run_coroutine_threadsafe
                asyncio.run_coroutine_threadsafe(
                    ws_manager.send_progress(task_id, formatted_message, mapped_progress), loop
                )

            # Build environment (blocking call in thread pool)
            await asyncio.to_thread(env_manager.get_or_create_env, manifest, env_progress)

        # Step 3: Pre-cache torch.hub repo for embedding models (if needed)
        if manifest.torch_hub_model:
            await ws_manager.send_progress(task_id, "Caching model architecture...", 0.95)

            python_path = env_manager.get_python(f"env-{manifest.env}")
            cache_cmd = [
                str(python_path), "-c",
                f"import torch; torch.hub.load('facebookresearch/dinov2', '{manifest.torch_hub_model}', pretrained=False)",
            ]

            import subprocess
            result = await asyncio.to_thread(
                subprocess.run, cache_cmd,
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(f"torch.hub cache warming failed: {result.stderr}")
                # Non-fatal — inference will still work if user has internet
            else:
                logger.info(f"Cached torch.hub architecture for {manifest.torch_hub_model}")

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
    updates = getattr(request.app.state, "model_updates", {
        "new_models": [],
        "checked_at": None
    })
    return updates


@router.get("/models/detection", response_model=list[ModelInfo])
def list_detection_models() -> list[ModelInfo]:
    """
    List all available detection models.

    Returns model metadata for UI dropdowns, sorted alphabetically by friendly_name.
    """
    manifest_mgr, _, _ = _get_managers()
    models = manifest_mgr.get_detection_models()

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

    # Add "None" option first
    result = [
        ModelInfo(
            model_id="none",
            friendly_name="No classification",
            emoji="⊘",
            type="classification",
            description="Run detection only, without species classification",
        )
    ]

    # Add actual classification models, sorted alphabetically
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
        )
        for manifest in models.values()
    ]

    # Sort alphabetically by friendly_name
    result.extend(sorted(model_list, key=lambda m: m.friendly_name))

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

    # Add "None" option first
    result = [
        ModelInfo(
            model_id="none",
            friendly_name="No embeddings",
            emoji="⊘",
            type="embedding",
            description="Skip embedding computation",
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
    from pathlib import Path
    from app.ml.taxonomy_parser import parse_taxonomy_csv, get_all_leaf_classes
    from app.core.config import get_settings

    settings = get_settings()
    manifest_mgr, _, _ = _get_managers()

    # Validate model exists
    try:
        manifest = manifest_mgr.get_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Only classification models have taxonomy
    if manifest.model_category == "detection":
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_id} is a detection model and does not have taxonomy"
        )

    # Find taxonomy.csv in model directory
    # Look in ~/AddaxAI/models/cls/{model_id}/taxonomy.csv
    taxonomy_path = settings.user_data_dir / "models" / "cls" / model_id / "taxonomy.csv"

    if not taxonomy_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Taxonomy file not found for model {model_id}. "
                   f"Expected at: {taxonomy_path}"
        )

    try:
        tree = parse_taxonomy_csv(taxonomy_path)
        all_classes = get_all_leaf_classes(tree)

        return {
            "tree": tree,
            "all_classes": all_classes
        }
    except Exception as e:
        logger.error(f"Failed to parse taxonomy for {model_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse taxonomy: {str(e)}"
        )


@router.get("/models/speciesnet/locations")
def get_speciesnet_locations():
    """
    Get available countries and US states for SpeciesNet geographic location selection.

    Returns dictionaries mapping display names (with emojis) to ISO codes.

    Returns:
        {
            "countries": {"🇺🇸 United States": "USA", ...},
            "us_states": {"🌴 California": "CA", ...}
        }
    """
    try:
        from app.ml.data.countries import countries_data, us_states_data

        return {
            "countries": countries_data,
            "us_states": us_states_data
        }
    except Exception as e:
        logger.error(f"Failed to load location data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load location data: {str(e)}"
        )
