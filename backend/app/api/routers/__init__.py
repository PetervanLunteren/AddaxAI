"""API routers."""

from .deployment_queue import router as deployment_queue_router
from .deployments import router as deployments_router
from .files import router as files_router
from .jobs import router as jobs_router
from .logs import router as logs_router
from .ml_models import router as ml_models_router
from .projects import router as projects_router
from .sites import router as sites_router
from .websocket import router as websocket_router

__all__ = [
    "deployment_queue_router",
    "deployments_router",
    "files_router",
    "jobs_router",
    "logs_router",
    "ml_models_router",
    "projects_router",
    "sites_router",
    "websocket_router",
]
