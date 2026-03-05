"""
WebSocket endpoints for real-time job progress.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
"""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging_config import get_logger
from app.core.websocket_manager import ws_manager
from app.db.base import get_db
from app.api.crud import job as job_crud
from app.api.crud import deployment_queue as queue_crud

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
        logger.info(f"Job {job_id} has in-memory state (type={ws_manager.current_state[job_id].get('type')}), skipping DB check")
    else:
        try:
            db = next(get_db())
            try:
                job = job_crud.get_job(db, job_id)
                if job and job.status == "completed":
                    await websocket.send_json({
                        "type": "complete",
                        "job_id": job_id,
                        "success": True,
                        "message": "Job completed (reconnected after restart)",
                        "data": {},
                    })
                elif job and job.status == "failed":
                    await websocket.send_json({
                        "type": "error",
                        "job_id": job_id,
                        "message": "Job failed (reconnected after restart)",
                    })
                elif job and job.status == "running":
                    # Job is "running" but no buffer exists — the worker
                    # died (e.g. backend was killed). Mark it failed in the
                    # DB and notify the client.
                    job_crud.update_job_status(db, job_id, "failed")
                    logger.warning(f"Job {job_id} was stuck in 'running' after restart, marked as failed")

                    # Also mark any associated queue entries as failed
                    payload = job.payload or {}
                    error_msg = "Interrupted by server restart"
                    queue_entry_ids = payload.get("queue_entry_ids", [])
                    if not queue_entry_ids and payload.get("queue_entry_id"):
                        queue_entry_ids = [payload["queue_entry_id"]]
                    for entry_id in queue_entry_ids:
                        entry = queue_crud.get_queue_entry(db, entry_id)
                        if entry and entry.status == "processing":
                            queue_crud.update_queue_status(db, entry_id, status="failed", error=error_msg)
                            logger.warning(f"Queue entry {entry_id} marked as failed (server restart)")

                    await websocket.send_json({
                        "type": "error",
                        "job_id": job_id,
                        "message": "Job was interrupted by server restart",
                    })
            finally:
                db.close()
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
                    if msg.get("type") == "ready":
                        await ws_manager.handle_ready(job_id)
                except (json.JSONDecodeError, AttributeError):
                    pass  # Ignore malformed messages

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, job_id)
        logger.info(f"Client disconnected from job {job_id}")

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        await ws_manager.disconnect(websocket, job_id)
