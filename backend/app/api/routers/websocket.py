"""
WebSocket endpoints for real-time job progress.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for job progress updates.

    Clients connect to this endpoint to receive real-time updates
    about job progress, completion, and errors.

    Args:
        websocket: WebSocket connection
        job_id: Job ID to subscribe to

    Message format (sent to client):
        {
            "type": "progress" | "complete" | "error",
            "job_id": str,
            "message": str,
            "progress": float (0.0-1.0, only for progress),
            "success": bool (only for complete),
            "data": dict (optional)
        }
    """
    await ws_manager.connect(websocket, job_id)

    try:
        # Keep connection alive until client disconnects
        while True:
            # Wait for any client messages (mostly for keep-alive)
            await websocket.receive_text()

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, job_id)
        logger.info(f"Client disconnected from job {job_id}")

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        await ws_manager.disconnect(websocket, job_id)
