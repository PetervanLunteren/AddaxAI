"""
WebSocket manager for real-time job progress updates.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Crash early if setup fails
- Explicit error handling
"""

import asyncio
from typing import Any, Callable

from fastapi import WebSocket

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for job progress updates.

    Uses a ready-handshake protocol: API endpoints register a start function
    via register_start(), and the actual work only begins when the frontend
    sends {"type": "ready"} over the WebSocket. This eliminates race
    conditions without buffers or artificial delays.
    """

    def __init__(self):
        # Map job_id -> list of WebSocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}
        # Latest progress/complete/error message per task (for reconnection)
        self.current_state: dict[str, dict] = {}
        # Registered start functions waiting for "ready" signal
        self._pending_starts: dict[str, Callable] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """
        Accept and register a WebSocket connection for a job.
        Sends current state if available (handles reconnections mid-job).
        """
        await websocket.accept()

        async with self._lock:
            if job_id not in self.active_connections:
                self.active_connections[job_id] = []
            self.active_connections[job_id].append(websocket)

        logger.info(f"WebSocket connected for job {job_id}")

        # Send current state to catch up reconnecting clients
        async with self._lock:
            state = self.current_state.get(job_id)

        if state:
            try:
                await websocket.send_json(state)
                logger.info(
                    f"Sent current state to reconnecting client for job {job_id}: "
                    f"type={state['type']}, message={state.get('message', '')[:50]}"
                )
            except Exception as e:
                logger.warning(f"Failed to send current state: {e}")
        else:
            logger.info(f"No current state for job {job_id}")

    async def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if job_id in self.active_connections:
                self.active_connections[job_id].remove(websocket)

                # Clean up empty job lists
                if not self.active_connections[job_id]:
                    del self.active_connections[job_id]

        logger.info(f"WebSocket disconnected for job {job_id}")

    def register_start(self, task_id: str, start_fn: Callable) -> None:
        """
        Register a worker coroutine to start when the frontend sends "ready".

        Called by API endpoints instead of asyncio.create_task(). The start_fn
        must be an async callable (coroutine function or lambda returning coroutine).
        """
        self._pending_starts[task_id] = start_fn
        logger.info(f"Registered pending start for task {task_id}")

    async def handle_ready(self, task_id: str) -> None:
        """
        Handle "ready" signal from frontend. Pops and starts the registered worker.

        Idempotent: no-op on reconnection (start_fn already popped on first call).
        """
        start_fn = self._pending_starts.pop(task_id, None)
        if start_fn is not None:
            logger.info(f"Received 'ready' for task {task_id}, starting worker")
            asyncio.create_task(start_fn())
        else:
            logger.debug(f"Received 'ready' for task {task_id}, but no pending start (reconnection or already started)")

    async def send_progress(
        self,
        job_id: str,
        message: str,
        progress: float,
        phase: str | None = None,
        phase_progress: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Send progress update to all clients subscribed to a job.
        Stores as current state (replaces previous, not append).
        """
        progress_data = {
            "type": "progress",
            "job_id": job_id,
            "message": message,
            "progress": progress,
            "phase": phase,
            "phase_progress": phase_progress,
            "data": data or {},
        }

        # Store as current state (latest only)
        async with self._lock:
            self.current_state[job_id] = progress_data

        # Send to all currently connected clients
        if job_id in self.active_connections:
            disconnected: list[WebSocket] = []

            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(progress_data)
                    # Force event loop to yield and flush WebSocket buffer
                    await asyncio.sleep(0)
                except Exception as e:
                    logger.warning(f"Failed to send progress to client: {e}")
                    disconnected.append(connection)

            # Clean up disconnected clients
            if disconnected:
                async with self._lock:
                    for connection in disconnected:
                        if connection in self.active_connections[job_id]:
                            self.active_connections[job_id].remove(connection)

    async def send_complete(
        self, job_id: str, success: bool, message: str, data: dict[str, Any] | None = None
    ) -> None:
        """
        Send completion message to all clients subscribed to a job.
        Stores in current state and schedules cleanup.
        """
        logger.info(f"send_complete() called for job {job_id}: success={success}, message={message[:50]}")

        complete_data = {
            "type": "complete",
            "job_id": job_id,
            "success": success,
            "message": message,
            "data": data or {},
        }

        # Store and send atomically
        async with self._lock:
            self.current_state[job_id] = complete_data

            if job_id in self.active_connections:
                logger.info(f"Sending completion to {len(self.active_connections[job_id])} connected clients")
                connections_to_send = list(self.active_connections[job_id])
            else:
                logger.info(f"No active connections for job {job_id}, completion stored in state only")
                connections_to_send = []

        for connection in connections_to_send:
            try:
                await connection.send_json(complete_data)
                await asyncio.sleep(0)
                logger.info("Successfully sent completion message to client")
            except Exception as e:
                logger.warning(f"Failed to send completion to client: {e}")
                async with self._lock:
                    if job_id in self.active_connections and connection in self.active_connections[job_id]:
                        self.active_connections[job_id].remove(connection)

        # Clean up state after 60s (client has long since received it)
        asyncio.create_task(self._cleanup_state(job_id, delay=60))

    async def send_error(self, job_id: str, error: str) -> None:
        """
        Send error message to all clients subscribed to a job.
        Stores in current state and schedules cleanup.
        """
        logger.info(f"send_error() called for job {job_id}: error={error[:50]}")

        error_data = {
            "type": "error",
            "job_id": job_id,
            "message": error,
        }

        # Store and send atomically
        async with self._lock:
            self.current_state[job_id] = error_data

            if job_id in self.active_connections:
                logger.info(f"Sending error to {len(self.active_connections[job_id])} connected clients")
                connections_to_send = list(self.active_connections[job_id])
            else:
                logger.info(f"No active connections for job {job_id}, error stored in state only")
                connections_to_send = []

        for connection in connections_to_send:
            try:
                await connection.send_json(error_data)
                await asyncio.sleep(0)
                logger.info("Successfully sent error message to client")
            except Exception as e:
                logger.warning(f"Failed to send error to client: {e}")
                async with self._lock:
                    if job_id in self.active_connections and connection in self.active_connections[job_id]:
                        self.active_connections[job_id].remove(connection)

        # Clean up state after 60s
        asyncio.create_task(self._cleanup_state(job_id, delay=60))

    def get_connection_count(self, job_id: str) -> int:
        """Get number of active connections for a job."""
        return len(self.active_connections.get(job_id, []))

    async def _cleanup_state(self, job_id: str, delay: int = 60) -> None:
        """Clean up current state for a job after a delay."""
        await asyncio.sleep(delay)
        if job_id in self.current_state:
            del self.current_state[job_id]
            logger.debug(f"Cleaned up state for job {job_id}")


# Global connection manager instance
ws_manager = ConnectionManager()
