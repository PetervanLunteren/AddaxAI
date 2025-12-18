"""
ML Models API endpoints for status checking and preparation.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
"""

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, status
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

        # Step 1: Download weights from HuggingFace
        # Get the current event loop for use in thread callbacks
        loop = asyncio.get_running_loop()

        await ws_manager.send_progress(
            task_id, "Downloading model weights from HuggingFace...", 0.1
        )

        def weights_progress(message: str, progress: float):
            """Sync callback for weight download progress."""
            # Map to 0.1-0.5 range
            mapped_progress = 0.1 + (progress * 0.4)
            # Schedule coroutine from thread using run_coroutine_threadsafe
            asyncio.run_coroutine_threadsafe(
                ws_manager.send_progress(task_id, message, mapped_progress), loop
            )

        # Download weights (blocking call in thread pool)
        # All models (detection + classification) now download from HF
        await asyncio.to_thread(
            model_storage.download_weights, manifest, weights_progress
        )

        await ws_manager.send_progress(task_id, "Weights downloaded", 0.5)

        # Step 2: Build environment
        await ws_manager.send_progress(task_id, "Building environment...", 0.6)

        def env_progress(message: str, progress: float):
            """Sync callback for environment build progress."""
            # Map to 0.6-0.95 range
            mapped_progress = 0.6 + (progress * 0.35)
            logger.info(f"Environment progress: {message} ({progress:.1%} -> {mapped_progress:.1%})")
            # Schedule coroutine from thread using run_coroutine_threadsafe
            asyncio.run_coroutine_threadsafe(
                ws_manager.send_progress(task_id, message, mapped_progress), loop
            )

        # Build environment (blocking call in thread pool)
        await asyncio.to_thread(env_manager.get_or_create_env, manifest, env_progress)

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
