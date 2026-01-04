"""
WebSocket manager for real-time job progress updates.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Crash early if setup fails
- Explicit error handling
"""

import asyncio
from collections import deque
from typing import Any

from fastapi import WebSocket

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for job progress updates.

    Allows multiple clients to subscribe to updates for specific jobs.
    Buffers recent messages to handle race conditions where connections
    arrive after progress updates have been sent.
    """

    def __init__(self, buffer_size: int = 50):
        """
        Initialize connection manager.

        Args:
            buffer_size: Number of recent messages to buffer per job
        """
        # Map job_id -> list of WebSocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}
        # Map job_id -> deque of recent messages (for replay to late connections)
        self.message_buffer: dict[str, deque[dict[str, Any]]] = {}
        self.buffer_size = buffer_size
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """
        Accept and register a WebSocket connection for a job.
        Replays any buffered messages to catch up the new connection.

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

        # Replay buffered messages to catch up this connection
        if job_id in self.message_buffer:
            buffer_count = len(self.message_buffer[job_id])
            logger.info(f"Replaying {buffer_count} buffered messages for job {job_id}")
            for buffered_msg in self.message_buffer[job_id]:
                try:
                    await websocket.send_json(buffered_msg)
                    progress_info = f", progress={buffered_msg.get('progress', 'N/A')}" if 'progress' in buffered_msg else ""
                    logger.info(f"Replayed: {buffered_msg['type']}{progress_info}, message={buffered_msg.get('message', '')[:50]}")
                except Exception as e:
                    logger.warning(f"Failed to replay buffered message: {e}")
        else:
            logger.info(f"No buffered messages for job {job_id}")

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
        Buffers the message for late-arriving connections.

        Args:
            job_id: Job ID
            message: Progress message
            progress: Progress value (0.0-1.0)
            data: Optional additional data
        """
        # Build progress message
        progress_data = {
            "type": "progress",
            "job_id": job_id,
            "message": message,
            "progress": progress,
            "data": data or {},
        }

        # Buffer the message for late-arriving connections
        if job_id not in self.message_buffer:
            self.message_buffer[job_id] = deque(maxlen=self.buffer_size)
        self.message_buffer[job_id].append(progress_data)

        # Send to all currently connected clients
        if job_id in self.active_connections:
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
        Buffers the completion message and cleans up after a delay.

        Args:
            job_id: Job ID
            success: Whether job completed successfully
            message: Completion message
            data: Optional result data
        """
        # Build completion message
        complete_data = {
            "type": "complete",
            "job_id": job_id,
            "success": success,
            "message": message,
            "data": data or {},
        }

        # Buffer the completion message for late-arriving connections
        if job_id not in self.message_buffer:
            self.message_buffer[job_id] = deque(maxlen=self.buffer_size)
        self.message_buffer[job_id].append(complete_data)

        # Send to all currently connected clients
        if job_id in self.active_connections:
            for connection in list(self.active_connections[job_id]):
                try:
                    await connection.send_json(complete_data)
                except Exception as e:
                    logger.warning(f"Failed to send completion to client: {e}")

            # Close all connections for this job
            async with self._lock:
                if job_id in self.active_connections:
                    del self.active_connections[job_id]

        # Clean up buffer after 60 seconds (allows late connections to still get completion)
        asyncio.create_task(self._cleanup_buffer(job_id, delay=60))

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

    async def _cleanup_buffer(self, job_id: str, delay: int = 60) -> None:
        """
        Clean up buffered messages for a job after a delay.

        Args:
            job_id: Job ID
            delay: Delay in seconds before cleanup
        """
        await asyncio.sleep(delay)
        if job_id in self.message_buffer:
            del self.message_buffer[job_id]
            logger.debug(f"Cleaned up message buffer for job {job_id}")


# Global connection manager instance
ws_manager = ConnectionManager()
