"""
WebSocket manager for real-time job progress updates.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Crash early if setup fails
- Explicit error handling
"""

import asyncio
from typing import Any

from fastapi import WebSocket

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for job progress updates.

    Allows multiple clients to subscribe to updates for specific jobs.
    """

    def __init__(self):
        """Initialize connection manager."""
        # Map job_id -> list of WebSocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """
        Accept and register a WebSocket connection for a job.

        Args:
            websocket: WebSocket connection
            job_id: Job ID to subscribe to
        """
        await websocket.accept()

        async with self._lock:
            if job_id not in self.active_connections:
                self.active_connections[job_id] = []
            self.active_connections[job_id].append(websocket)

        logger.info(f"WebSocket connected for job {job_id}")

    async def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
            job_id: Job ID
        """
        async with self._lock:
            if job_id in self.active_connections:
                self.active_connections[job_id].remove(websocket)

                # Clean up empty job lists
                if not self.active_connections[job_id]:
                    del self.active_connections[job_id]

        logger.info(f"WebSocket disconnected for job {job_id}")

    async def send_progress(
        self, job_id: str, message: str, progress: float, data: dict[str, Any] | None = None
    ) -> None:
        """
        Send progress update to all clients subscribed to a job.

        Args:
            job_id: Job ID
            message: Progress message
            progress: Progress value (0.0-1.0)
            data: Optional additional data
        """
        if job_id not in self.active_connections:
            return  # No clients connected for this job

        # Build progress message
        progress_data = {
            "type": "progress",
            "job_id": job_id,
            "message": message,
            "progress": progress,
            "data": data or {},
        }

        # Send to all connected clients
        disconnected: list[WebSocket] = []

        for connection in self.active_connections[job_id]:
            try:
                await connection.send_json(progress_data)
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

        Args:
            job_id: Job ID
            success: Whether job completed successfully
            message: Completion message
            data: Optional result data
        """
        if job_id not in self.active_connections:
            return

        # Build completion message
        complete_data = {
            "type": "complete",
            "job_id": job_id,
            "success": success,
            "message": message,
            "data": data or {},
        }

        # Send to all connected clients
        for connection in list(self.active_connections[job_id]):
            try:
                await connection.send_json(complete_data)
            except Exception as e:
                logger.warning(f"Failed to send completion to client: {e}")

        # Close all connections for this job
        async with self._lock:
            if job_id in self.active_connections:
                del self.active_connections[job_id]

    async def send_error(self, job_id: str, error: str) -> None:
        """
        Send error message to all clients subscribed to a job.

        Args:
            job_id: Job ID
            error: Error message
        """
        if job_id not in self.active_connections:
            return

        # Build error message
        error_data = {
            "type": "error",
            "job_id": job_id,
            "error": error,
        }

        # Send to all connected clients
        for connection in list(self.active_connections[job_id]):
            try:
                await connection.send_json(error_data)
            except Exception as e:
                logger.warning(f"Failed to send error to client: {e}")

    def get_connection_count(self, job_id: str) -> int:
        """
        Get number of active connections for a job.

        Args:
            job_id: Job ID

        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(job_id, []))


# Global connection manager instance
ws_manager = ConnectionManager()
