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

from app import __version__
from app.api.routers import (
    backup_router,
    deployment_queue_router,
    deployments_router,
    detections_router,
    events_router,
    export_router,
    files_router,
    folder_runs_router,
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
        app.state.model_updates = {
            "new_models": [],
            "refreshed_models": [],
            "drifted_models": [],
            "drifted_envs": [],
            "checked_at": None,
            "disabled": True,
        }
        return

    try:
        updater = ModelCatalogUpdater(catalog_url=settings.model_catalog_url)
        result = await updater.sync()
        app.state.model_updates = result
    except Exception as e:
        logger.error(f"Model catalog sync failed: {e}", exc_info=True)
        app.state.model_updates = {
            "new_models": [],
            "refreshed_models": [],
            "drifted_models": [],
            "drifted_envs": [],
            "error": str(e),
        }


async def _warm_up_query_caches() -> None:
    """
    Pre-load commonly-queried tables and indexes into SQLite's page
    cache (and transitively the OS file cache) at startup. Without
    this, every "first" user interaction in a session pays the cold
    disk-read cost: opening the projects list, opening a project,
    rendering the dashboard, opening the taxonomy filter modal, etc.
    Each of those touches a different set of tables, so the slowness
    follows the user around for the first minute or two.

    By running representative queries here, the pages land in cache
    during the splash screen instead of during the user's first click.
    On a fresh OS boot or after a reboot the OS file cache is empty;
    this task is what makes the difference between "first launch is
    slow" and "first launch feels normal".

    Non-blocking, non-fatal: a perf optimisation, not a correctness
    step. Tables that don't exist yet (rare; alembic ran in init_db
    immediately above) are tolerated.
    """
    from sqlalchemy import text

    from app.db.base import get_session_factory

    # Each query exercises a different hot path. The COUNT(*) calls
    # force a primary-key index scan on each table; the JOIN counts
    # also pull the foreign-key indexes into cache. Together these
    # cover the dashboard, verification grid, taxonomy modal, project
    # list, deployment list, and the events / observations pages.
    warmup_sql = (
        "SELECT COUNT(*) FROM projects",
        "SELECT COUNT(*) FROM sites",
        "SELECT COUNT(*) FROM deployments",
        "SELECT COUNT(*) FROM files",
        "SELECT COUNT(*) FROM detections",
        "SELECT COUNT(*) FROM events",
        "SELECT COUNT(*) FROM event_observations",
        "SELECT COUNT(*) FROM label_taxonomy",
        "SELECT COUNT(*) FROM jobs",
        "SELECT COUNT(*) FROM deployment_queue",
        # Cross-table joins that pull foreign-key indexes into cache.
        "SELECT COUNT(*) FROM detections d JOIN files f ON d.file_id = f.id",
        "SELECT COUNT(*) FROM event_observations o JOIN events e ON o.event_id = e.id",
        "SELECT COUNT(*) FROM detections d LEFT JOIN label_taxonomy lt ON d.label = lt.name",
        "SELECT COUNT(*) FROM files f JOIN deployments d ON f.deployment_id = d.id",
        "SELECT COUNT(*) FROM events e JOIN deployments d ON e.deployment_id = d.id",
        # Relabel detection picker: opens a search list of custom
        # labels filtered by project. Hits the (project_id, is_custom)
        # path on label_taxonomy.
        "SELECT COUNT(*) FROM label_taxonomy WHERE is_custom = 1",
        # Deployment / site info slideouts: top species panel joins
        # event observations to events, then left-joins label_taxonomy
        # for scientific_name. Same path the dashboard's species charts
        # use too.
        "SELECT COUNT(*) FROM event_observations o "
        "JOIN events e ON o.event_id = e.id "
        "LEFT JOIN label_taxonomy lt ON lt.name = o.label",
        # Slideouts compute file totals + verified counts per
        # deployment / site. Reads file_type, verified, size_bytes
        # off the files row.
        "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM files",
    )

    def _run() -> None:
        session_factory = get_session_factory()
        db = session_factory()
        try:
            for sql in warmup_sql:
                try:
                    db.execute(text(sql)).scalar()
                except Exception as e:
                    # A missing table on a partially-migrated DB shouldn't
                    # break the whole warm-up. Log and move on.
                    logger.debug(f"Warm-up query skipped ({sql!r}): {e}")
        finally:
            db.close()

    try:
        await asyncio.to_thread(_run)
        logger.info("Database warm-up complete")
    except Exception as e:
        logger.warning(f"Database warm-up failed: {e}")


async def _reclaim_legacy_video_frames() -> None:
    """
    Delete legacy bulk-extracted video frame JPEGs left over from the
    pre-2026-05 pipeline. The 2026-05 refactor stopped writing per-frame
    JPEGs to disk (only the single best frame per video survives), and
    the matching `file_type='frame'` DB rows are collapsed by the
    alembic migration `drop_frame_file_rows`. This task removes the
    JPEGs the migration intentionally doesn't touch (alembic stays out
    of the filesystem).

    For each video File row, the only JPEG we keep under
    `<deployment>/.addaxai/projects/*/video_frames/<rel_video>/` is the
    one named `frame{best_frame_number:06d}.jpg`. Everything else gets
    unlinked. Empty subdirectories are tolerated; the function never
    creates files, so re-running it on a clean tree is a no-op.

    Best-effort and non-fatal: a slow / unmounted drive, a permission
    error, or a removed deployment folder all surface as a warning and
    move on. We never block startup on this.
    """
    from app.db.base import get_session_factory
    from app.models import File

    def _run() -> dict[str, int]:
        session_factory = get_session_factory()
        db = session_factory()
        reclaimed_bytes = 0
        deleted = 0
        scanned = 0
        try:
            videos = (
                db.query(File)
                .filter(File.file_type == "video")
                .all()
            )
            # Group by deployment.folder_path so we only stat each
            # deployment's video_frames tree once.
            keep_filenames: dict[Path, set[Path]] = {}
            for v in videos:
                if not v.best_frame_path:
                    continue
                bf = Path(v.best_frame_path)
                # The best-frame JPEG lives inside a deployment's
                # `.addaxai/projects/<pid>/video_frames/<rel_video>/`.
                # `bf.parent` is that per-video directory.
                keep_filenames.setdefault(bf.parent, set()).add(bf.name)

            for video_dir, keep_names in keep_filenames.items():
                if not video_dir.exists():
                    continue
                scanned += 1
                for jpeg in video_dir.glob("frame*.jpg"):
                    if jpeg.name in keep_names:
                        continue
                    try:
                        size = jpeg.stat().st_size
                        jpeg.unlink()
                        reclaimed_bytes += size
                        deleted += 1
                    except OSError as e:
                        logger.warning(
                            f"Could not delete legacy frame {jpeg}: {e}"
                        )
            return {
                "scanned_video_dirs": scanned,
                "deleted_jpegs": deleted,
                "reclaimed_bytes": reclaimed_bytes,
            }
        finally:
            db.close()

    try:
        result = await asyncio.to_thread(_run)
        if result["deleted_jpegs"]:
            mb = result["reclaimed_bytes"] / (1024 * 1024)
            logger.info(
                f"Legacy video-frame cleanup: "
                f"removed {result['deleted_jpegs']} JPEG(s) "
                f"across {result['scanned_video_dirs']} video dir(s), "
                f"reclaimed {mb:.1f} MB"
            )
        else:
            logger.debug(
                f"Legacy video-frame cleanup: "
                f"nothing to reclaim across "
                f"{result['scanned_video_dirs']} video dir(s)"
            )
    except Exception as e:
        logger.error(f"Legacy video-frame cleanup failed: {e}", exc_info=True)


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

    # Honour a pending "restore DB from backup" request. Done BEFORE
    # init_db so the alembic upgrade in init_db sees the restored DB.
    # The marker is consumed unconditionally (even on validation failure)
    # so a corrupt request can't loop the user through restore-fail-
    # restore-fail forever. The current live DB is force-snapshotted to
    # the ring buffer first as a safety net.
    from app.db.backup import consume_restore_marker
    consume_restore_marker(settings)

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

    # Pre-upgrade backup (best-effort; never blocks startup). Must run
    # BEFORE init_db so the snapshot captures the pre-migration schema.
    # Skipped on a fresh install: there is no existing DB to preserve.
    live_db = settings.user_data_dir / "addaxai.db"
    if live_db.is_file():
        from app.db.backup import pre_upgrade_backup
        from app.db.base import get_engine
        from app.db.migrations import get_current_revision, needs_upgrade
        try:
            engine = get_engine()
            if needs_upgrade(engine):
                pre_upgrade_backup(settings, rev=get_current_revision(engine))
        except Exception as e:
            logger.error(f"Pre-upgrade backup failed: {e}", exc_info=True)

    # Initialize database - will crash if it fails
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise

    # Fail jobs left `running` by a previous process. Analysis runs in an
    # in-memory worker, so a restart or crash orphans the job row and its
    # queue entries as perpetually in-progress. Nothing is processing them
    # now, so mark them failed before we start serving. Best-effort.
    try:
        from app.api.crud.job import reconcile_interrupted_jobs
        from app.db.base import get_session_factory

        with get_session_factory()() as db:
            n = reconcile_interrupted_jobs(db)
        if n:
            logger.warning(
                f"Marked {n} interrupted job(s) as failed (left running by a "
                f"previous process)"
            )
    except Exception as e:
        logger.error(f"Job reconciliation failed: {e}", exc_info=True)

    # Daily rolling backup (best-effort). Runs AFTER init_db so a fresh
    # install also produces a snapshot on first launch — at this point
    # the DB exists either way (init_db creates it on fresh installs and
    # opens it on subsequent launches). Throttled to one per UTC day, so
    # rapid restarts do not clobber the ring buffer.
    if live_db.is_file():
        from app.db.backup import ring_buffer_backup
        try:
            ring_buffer_backup(settings)
        except Exception as e:
            logger.error(f"Daily backup failed: {e}", exc_info=True)

    # Start background tasks (non-blocking)
    sync_task = asyncio.create_task(update_model_catalog(app))
    thumbnail_task = asyncio.create_task(auto_generate_thumbnails())
    folder_check_task = asyncio.create_task(_check_deployment_folders_on_startup())
    warmup_task = asyncio.create_task(_warm_up_query_caches())
    legacy_frame_cleanup_task = asyncio.create_task(_reclaim_legacy_video_frames())

    yield

    # Shutdown: cancel background tasks if still running
    for task in (
        sync_task,
        thumbnail_task,
        folder_check_task,
        warmup_task,
        legacy_frame_cleanup_task,
    ):
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
        version=__version__,
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Unhandled-exception middleware. Without this, an unhandled
    # exception flows up to starlette's default ServerErrorMiddleware,
    # which sits OUTSIDE CORSMiddleware in the stack. The 500 it
    # generates ships without `Access-Control-Allow-Origin`, and the
    # renderer (which treats `localhost:8000` and `127.0.0.1:8000` as
    # cross-origin) reports the response as `TypeError: Failed to
    # fetch`, hiding the real status code and making diagnostics
    # painful. Registering a FastAPI `@app.exception_handler(Exception)`
    # does NOT fix this: Starlette's build_middleware_stack routes
    # Exception/500 handlers to ServerErrorMiddleware (also outermost).
    #
    # The middleware below must be added BEFORE CORSMiddleware.
    # add_middleware inserts each new entry at the front of
    # `user_middleware`, then the stack is built by wrapping that list
    # in reverse, so the first-registered middleware ends up innermost.
    # We want exception-handling to be inside CORS so the 500 response
    # it produces gets the CORS header on the way out.
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def unhandled_exception_middleware(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logger.error(
                f"Unhandled exception on {request.method} {request.url.path}: {exc}",
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error"},
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
    app.include_router(backup_router)
    app.include_router(setup_router)
    app.include_router(sites_router)
    app.include_router(deployments_router)
    app.include_router(deployment_queue_router)
    app.include_router(detections_router)
    app.include_router(events_router)
    app.include_router(export_router)
    app.include_router(files_router)
    app.include_router(folder_runs_router)
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
            "version": __version__,
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
                "version": __version__,
                "docs": "/docs",
                "health": "/health",
                "note": "Frontend not available - build frontend and bundle with PyInstaller",
            }

    return app


# Create app instance
app = create_app()
