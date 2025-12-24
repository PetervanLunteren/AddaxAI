"""
FastAPI main application entry point.

Following DEVELOPERS.md principles:
- Crash early if configuration is missing
- Explicit setup, no silent defaults
- Type hints everywhere
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from app.api.routers import (
    deployment_queue_router,
    deployments_router,
    files_router,
    jobs_router,
    logs_router,
    ml_models_router,
    projects_router,
    sites_router,
    websocket_router,
)
from app.core.config import get_settings
from app.core.logging_config import get_logger, setup_logging
from app.db.base import init_db
from app.ml.catalog_updater import ModelCatalogUpdater

# Initialize logging first, before anything else
setup_logging()
logger = get_logger(__name__)


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

    # Initialize database - will crash if it fails
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise

    # Start model catalog sync in background (non-blocking)
    sync_task = asyncio.create_task(update_model_catalog(app))

    yield

    # Shutdown
    # Cancel catalog sync if still running
    if not sync_task.done():
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass

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
    app.include_router(sites_router)
    app.include_router(deployments_router)
    app.include_router(deployment_queue_router)
    app.include_router(files_router)
    app.include_router(jobs_router)
    app.include_router(logs_router)
    app.include_router(ml_models_router)
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

        # Serve vite.svg favicon
        @app.get("/vite.svg")
        def serve_vite_svg():
            return FileResponse(str(frontend_dir / "vite.svg"))

        # Catch-all route for SPA - serve index.html for all frontend routes
        # This must be last to not override API routes
        @app.get("/{full_path:path}")
        def serve_frontend(full_path: str):
            """
            Serve React frontend for all routes not handled by API.

            This enables client-side routing for the SPA.
            """
            # If path looks like a file request, try to serve it
            file_path = frontend_dir / full_path
            if file_path.is_file():
                return FileResponse(str(file_path))

            # Otherwise, serve index.html for SPA routing
            return FileResponse(str(frontend_dir / "index.html"))
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
