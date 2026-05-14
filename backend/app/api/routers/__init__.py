"""API routers."""

from .backup import router as backup_router
from .deployment_queue import router as deployment_queue_router
from .deployments import router as deployments_router
from .detections import router as detections_router
from .events import router as events_router
from .export import router as export_router
from .files import router as files_router
from .folder_runs import router as folder_runs_router
from .jobs import router as jobs_router
from .logs import router as logs_router
from .ml_models import router as ml_models_router
from .observations import router as observations_router
from .projects import router as projects_router
from .setup import router as setup_router
from .sites import router as sites_router
from .statistics import router as statistics_router
from .timelapse import router as timelapse_router
from .websocket import router as websocket_router

__all__ = [
    "backup_router",
    "deployment_queue_router",
    "deployments_router",
    "detections_router",
    "events_router",
    "export_router",
    "files_router",
    "folder_runs_router",
    "jobs_router",
    "logs_router",
    "ml_models_router",
    "observations_router",
    "projects_router",
    "setup_router",
    "sites_router",
    "statistics_router",
    "timelapse_router",
    "websocket_router",
]
