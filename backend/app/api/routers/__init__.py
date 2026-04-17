"""API routers."""

from .deployment_queue import router as deployment_queue_router
from .deployments import router as deployments_router
from .detections import router as detections_router
from .events import router as events_router
from .export import router as export_router
from .files import router as files_router
from .jobs import router as jobs_router
from .logs import router as logs_router
from .ml_models import router as ml_models_router
from .projects import router as projects_router
from .similarity import router as similarity_router
from .sites import router as sites_router
from .statistics import router as statistics_router
from .websocket import router as websocket_router

__all__ = [
    "deployment_queue_router",
    "deployments_router",
    "detections_router",
    "events_router",
    "export_router",
    "files_router",
    "jobs_router",
    "logs_router",
    "ml_models_router",
    "projects_router",
    "similarity_router",
    "sites_router",
    "statistics_router",
    "websocket_router",
]
