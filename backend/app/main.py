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

from app.core.config import get_settings
from app.db.base import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    Crashes if database initialization fails (following "crash early" principle).
    """
    # Startup
    settings = get_settings()
    print(f"Starting AddaxAI Backend (Environment: {settings.environment})")
    print(f"Database: {settings.database_url}")
    print(f"User data directory: {settings.user_data_dir}")

    # Initialize database - will crash if it fails
    init_db()
    print("Database initialized successfully")

    yield

    # Shutdown
    print("Shutting down AddaxAI Backend")


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
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
