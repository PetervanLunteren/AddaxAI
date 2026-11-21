"""
WebSocket endpoints for real-time job progress.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
"""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app.api.crud import deployment_queue as queue_crud
from app.api.crud import job as job_crud
from app.core.job_cancellation import request_cancel
from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db

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

    # If no in-memory state exists, check if the job already has a terminal
    # status in the DB. This handles backend restarts where state is lost.
    if job_id in ws_manager.current_state:
        state_type = ws_manager.current_state[job_id].get('type')
        logger.info(
            f"Job {job_id} has in-memory state "
            f"(type={state_type}), skipping DB check"
        )
    else:
        try:
            message = await run_in_threadpool(_settle_job_after_restart, job_id)
            if message is not None:
                await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to check job status for {job_id}: {e}")

    try:
        # Keep connection alive and handle client messages
        while True:
            text = await websocket.receive_text()
            # Parse JSON messages; plain "ping" is keepalive (ignored)
            if text and text.strip() != "ping":
                try:
                    msg = json.loads(text)
                    msg_type = msg.get("type")
                    if msg_type == "ready":
                        await ws_manager.handle_ready(job_id)
                    elif msg_type == "cancel":
                        logger.info(f"Cancel requested for job {job_id}")
                        request_cancel(job_id)
                except (json.JSONDecodeError, AttributeError):
                    pass  # Ignore malformed messages

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, job_id)
        logger.info(f"Client disconnected from job {job_id}")

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        await ws_manager.disconnect(websocket, job_id)


def _settle_job_after_restart(job_id: str) -> dict | None:
    """The message for a client that reconnects to a job this process has
    no state for, or None. A job still marked running had its worker die
    with the previous process, so it and its queue entries are marked
    failed here. Database work, so the WebSocket runs it in a thread."""
    db = next(get_db())
    try:
        job = job_crud.get_job(db, job_id)
        if job and job.status == "completed":
            return {
                "type": "complete",
                "job_id": job_id,
                "success": True,
                "message": "Job completed (reconnected after restart)",
                "data": {},
            }
        if job and job.status == "failed":
            return {
                "type": "error",
                "job_id": job_id,
                "message": "Job failed (reconnected after restart)",
            }
        if job and job.status == "running":
            job_crud.update_job_status(db, job_id, "failed")
            logger.warning(
                f"Job {job_id} was stuck in 'running' after restart, marked as failed"
            )
            payload = job.payload or {}
            error_msg = "Interrupted by server restart"
            queue_entry_ids = payload.get("queue_entry_ids", [])
            if not queue_entry_ids and payload.get("queue_entry_id"):
                queue_entry_ids = [payload["queue_entry_id"]]
            for entry_id in queue_entry_ids:
                entry = queue_crud.get_queue_entry(db, entry_id)
                if entry and entry.status == "processing":
                    queue_crud.update_queue_status(
                        db, entry_id, status="failed", error=error_msg
                    )
                    logger.warning(
                        f"Queue entry {entry_id} marked as failed (server restart)"
                    )
            return {
                "type": "error",
                "job_id": job_id,
                "message": "Job was interrupted by server restart",
            }
        return None
    finally:
        db.close()
