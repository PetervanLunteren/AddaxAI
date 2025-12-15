"""
Logging API endpoints.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/logs", tags=["Logging"])


class FrontendLogEntry(BaseModel):
    """Frontend log entry schema."""

    timestamp: str
    level: str  # "info" | "warn" | "error"
    message: str
    context: dict[str, object] | None = None


class FrontendLogsRequest(BaseModel):
    """Request body for forwarding frontend logs."""

    logs: list[FrontendLogEntry]


@router.post("", status_code=status.HTTP_201_CREATED)
def forward_frontend_logs(request: FrontendLogsRequest) -> dict[str, str]:
    """
    Forward frontend logs to backend log file.

    Receives batched logs from frontend and writes them to backend.log
    with [FRONTEND] prefix for easy filtering.

    Returns:
        Success message with count of logs received
    """
    frontend_logger = get_logger("frontend")

    for entry in request.logs:
        # Format: [FRONTEND] message {context}
        context_str = f" {entry.context}" if entry.context else ""
        log_message = f"[{entry.timestamp}] {entry.message}{context_str}"

        # Map frontend log levels to Python logging levels
        if entry.level == "error":
            frontend_logger.error(log_message)
        elif entry.level == "warn":
            frontend_logger.warning(log_message)
        else:  # info or anything else
            frontend_logger.info(log_message)

    logger.info(f"Received {len(request.logs)} frontend log entries")

    return {
        "status": "success",
        "message": f"Logged {len(request.logs)} entries",
    }
