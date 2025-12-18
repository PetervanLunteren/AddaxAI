"""
FastAPI main application entry point.

Following DEVELOPERS.md principles:
- Crash early if configuration is missing
- Explicit setup, no silent defaults
- Type hints everywhere
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import (
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

# Initialize logging first, before anything else
setup_logging()
logger = get_logger(__name__)


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

    yield

    # Shutdown
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
    # In production (Electron), frontend and backend are on same localhost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(projects_router)
    app.include_router(sites_router)
    app.include_router(deployments_router)
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
        }

    return app


# Create app instance
app = create_app()
