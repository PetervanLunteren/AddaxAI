"""
FastAPI main application entry point.

Following DEVELOPERS.md principles:
- Crash early if configuration is missing
- Explicit setup, no silent defaults
- Type hints everywhere
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from app.api.routers import (
    deployment_queue_router,
    deployments_router,
    detections_router,
    events_router,
    export_router,
    files_router,
    jobs_router,
    logs_router,
    ml_models_router,
    observations_router,
    projects_router,
    setup_router,
    sites_router,
    statistics_router,
    websocket_router,
)
from app.core.config import get_settings
from app.core.logging_config import get_logger, setup_logging
from app.db.base import init_db
from app.ml.catalog_updater import ModelCatalogUpdater

# Initialize logging first, before anything else
setup_logging()
logger = get_logger(__name__)


async def auto_generate_thumbnails() -> None:
    """Background task to auto-select thumbnails for projects without one.

    Runs non-blocking during startup. Projects that already have a
    thumbnail (user-uploaded or previously auto-selected) are skipped.
    """
    try:
        settings = get_settings()
        thumbnails_dir = settings.user_data_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)

        from app.db.base import get_session_factory
        from app.services.thumbnail_service import (
            auto_select_project_thumbnails,
        )

        def _run() -> None:
            session_factory = get_session_factory()
            db = session_factory()
            try:
                auto_select_project_thumbnails(db, thumbnails_dir)
            finally:
                db.close()

        await asyncio.to_thread(_run)
    except Exception as e:
        logger.error(
            f"Auto-thumbnail generation failed: {e}", exc_info=True
        )


async def update_model_catalog(app: FastAPI) -> None:
    """
    Background task to sync model catalog.

    Runs non-blocking during startup - app continues immediately.
    """
    settings = get_settings()

    # Skip if disabled
    if settings.disable_model_updates:
        logger.info("Model catalog updates disabled")
        app.state.model_updates = {"new_models": [], "checked_at": None, "disabled": True}
        return

    try:
        updater = ModelCatalogUpdater(catalog_url=settings.model_catalog_url)
        result = await updater.sync()
        app.state.model_updates = result
    except Exception as e:
        logger.error(f"Model catalog sync failed: {e}", exc_info=True)
        app.state.model_updates = {"new_models": [], "error": str(e)}


async def _check_deployment_folders_on_startup() -> None:
    """
    Re-stat every deployment's folder_path so the folder_status column
    reflects the current filesystem state. Runs as a non-blocking task
    during app startup, so slow or unmounted drives never block boot.
    """
    from app.api.crud.deployment import check_all_deployment_folders
    from app.db.base import get_session_factory

    try:
        # Run the sync DB work off the event loop so slow filesystem
        # stat calls don't block request handling.
        def _run() -> dict[str, int]:
            session_factory = get_session_factory()
            db = session_factory()
            try:
                return check_all_deployment_folders(db)
            finally:
                db.close()

        result = await asyncio.to_thread(_run)
        logger.info(
            f"Deployment folder check complete: "
            f"{result['checked']} checked, "
            f"{result['changed']} changed, "
            f"{result['skipped']} skipped"
        )
    except Exception as e:
        logger.error(f"Deployment folder check failed: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    Crashes if database initialization fails (following "crash early" principle).
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting AddaxAI Backend (Environment: {settings.environment})")
    logger.info(f"Database: {settings.database_url}")
    logger.info(f"User data directory: {settings.user_data_dir}")

    # Honour a pending "wipe DB on next launch" request from the reset
    # flow. We do this BEFORE init_db so no SQLAlchemy connection is
    # holding the file open. Marker is consumed (deleted) regardless of
    # whether DB files existed, to avoid an infinite loop on subsequent
    # launches.
    db_wipe_marker = settings.user_data_dir / ".wipe-db-on-next-launch"
    if db_wipe_marker.exists():
        logger.warning("DB wipe marker present — deleting addaxai.db files")
        for sibling in (
            "addaxai.db",
            "addaxai.db-wal",
            "addaxai.db-shm",
        ):
            target = settings.user_data_dir / sibling
            if target.exists():
                try:
                    target.unlink()
                    logger.warning(f"Removed {target}")
                except Exception as e:
                    logger.error(
                        f"Failed to remove {target}: {e}", exc_info=True
                    )
        try:
            db_wipe_marker.unlink()
        except Exception as e:
            logger.error(
                f"Failed to remove DB wipe marker: {e}", exc_info=True
            )

    # Initialize database - will crash if it fails
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise

    # Start background tasks (non-blocking)
    sync_task = asyncio.create_task(update_model_catalog(app))
    thumbnail_task = asyncio.create_task(auto_generate_thumbnails())
    folder_check_task = asyncio.create_task(_check_deployment_folders_on_startup())

    yield

    # Shutdown: cancel background tasks if still running
    for task in (sync_task, thumbnail_task, folder_check_task):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # Cancel pending WebSocket cleanup tasks
    from app.core.websocket_manager import ws_manager
    await ws_manager.close()

    logger.info("Shutting down AddaxAI Backend")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns configured FastAPI instance ready to serve.
    Will crash if settings are invalid (explicit configuration required).
    """
    settings = get_settings()

    app = FastAPI(
        title="AddaxAI API",
        description="Camera trap wildlife analysis platform - Backend API",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # CORS middleware - allow frontend to access API
    # In Electron: frontend and backend both served from port 8000 (same origin)
    # In dev: frontend on Vite dev server (5173), backend on 8000
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8000",  # Electron app (same origin)
            "http://127.0.0.1:8000",  # Electron app (same origin)
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers (already have /api prefix in their definitions)
    app.include_router(projects_router)
    app.include_router(setup_router)
    app.include_router(sites_router)
    app.include_router(deployments_router)
    app.include_router(deployment_queue_router)
    app.include_router(detections_router)
    app.include_router(events_router)
    app.include_router(export_router)
    app.include_router(files_router)
    app.include_router(jobs_router)
    app.include_router(logs_router)
    app.include_router(ml_models_router)
    app.include_router(observations_router)
    app.include_router(statistics_router)
    app.include_router(websocket_router)

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    def health_check() -> dict[str, str]:
        """
        Health check endpoint.

        Returns application status and version.
        """
        return {
            "status": "healthy",
            "version": "0.1.0",
            "environment": settings.environment,
        }

    # Get frontend static files directory
    # In development: frontend/dist from repo root
    # In production (PyInstaller): bundled with executable
    import sys
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        frontend_dir = Path(sys._MEIPASS) / "frontend" / "dist"
    else:
        # Running in development
        backend_dir = Path(__file__).parent.parent
        frontend_dir = backend_dir.parent / "frontend" / "dist"

    # Serve frontend static files if available
    if frontend_dir.exists():
        logger.info(f"Serving frontend from: {frontend_dir}")

        # Mount static assets directory
        assets_dir = frontend_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Catch-all route for SPA - serve index.html for all frontend routes
        # This must be last to not override API routes
        @app.get("/{full_path:path}")
        def serve_frontend(full_path: str):
            """
            Serve React frontend for all routes not handled by API.

            This enables client-side routing for the SPA.
            """
            # If path looks like a file request, try to serve it.
            # Hashed Vite assets (under /assets/, mounted above) get default
            # caching. Other static files at the root are also cacheable.
            file_path = frontend_dir / full_path
            if file_path.is_file():
                return FileResponse(str(file_path))

            # SPA entry point. Must never be cached: its URL is stable but it
            # references content-hashed assets that change every build. A
            # cached index.html on the user's machine after an upgrade points
            # at hashes that no longer exist, producing a white screen until
            # the user does a hard refresh.
            return FileResponse(
                str(frontend_dir / "index.html"),
                headers={"Cache-Control": "no-store"},
            )
    else:
        logger.warning(f"Frontend directory not found: {frontend_dir}")
        logger.warning("API will be available but frontend UI will not be served")

        # Fallback root endpoint showing API info
        @app.get("/", tags=["Root"])
        def root() -> dict[str, str]:
            """
            Root endpoint.

            Returns welcome message and API information.
            """
            return {
                "message": "AddaxAI API",
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/health",
                "note": "Frontend not available - build frontend and bundle with PyInstaller",
            }

    return app


# Create app instance
app = create_app()
