"""
WebSocket manager for real-time job progress updates.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Crash early if setup fails
- Explicit error handling
"""

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# How long (seconds) a registered start function can wait for a "ready" signal
# before being cleaned up. Prevents memory leaks if frontend never connects.
_PENDING_START_TIMEOUT = 300  # 5 minutes


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
        # Background cleanup tasks (tracked so they can be cancelled on shutdown)
        self._cleanup_tasks: set[asyncio.Task] = set()

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
            state = self.current_state.get(job_id)

        logger.info(f"WebSocket connected for job {job_id}")

        # Send current state to catch up reconnecting clients
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

        Schedules automatic cleanup after _PENDING_START_TIMEOUT seconds to
        prevent memory leaks if the frontend never connects.
        """
        # Drop any leftover terminal/progress state from a previous run that
        # reused this task_id. Model preparation keys on model_id, so the
        # same task_id recurs across runs; without this, a fresh prepare's
        # WebSocket replays the prior run's "cancelled"/"complete" the instant
        # it connects and the dialog closes immediately. Job runs use unique
        # ids and are unaffected.
        self.current_state.pop(task_id, None)
        self._pending_starts[task_id] = start_fn
        logger.info(f"Registered pending start for task {task_id}")

        # Schedule cleanup in case frontend never connects
        self._schedule_cleanup(self._cleanup_pending_start(task_id))

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
            logger.debug(
                f"Received 'ready' for task {task_id}, "
                f"but no pending start (reconnection or already started)"
            )

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

        # Store state and snapshot connections atomically
        async with self._lock:
            # compute_device is announced once per phase, by whichever
            # subprocess that phase runs, then never repeated. Every
            # later message in the phase needs it: the cached state is
            # what we replay to reconnecting clients, so without
            # forwarding it the UI slips back to "detecting..." after a
            # page reload.
            #
            # **Only within the same phase.** Each phase runs its own
            # model and resolves its own device, and the announcement
            # arrives seconds into the phase because it comes after the
            # model loads. Carrying the value across a phase boundary
            # fills that gap with the *previous* phase's hardware, which
            # is wrong the moment two models resolve differently: a
            # classifier with no GPU support falls back to CPU while
            # MegaDetector runs on the GPU, and the row would still say
            # GPU. A phase that never announces shows "detecting..."
            # forever, which is the honest answer and the reason phases
            # doing their own CPU work say so explicitly.
            prev = self.current_state.get(job_id)
            if (
                prev
                and prev.get("phase") == phase
                and prev.get("data", {}).get("compute_device")
                and not progress_data["data"].get("compute_device")
            ):
                progress_data["data"]["compute_device"] = prev["data"][
                    "compute_device"
                ]
            self.current_state[job_id] = progress_data

            if job_id in self.active_connections:
                connections_to_send = list(self.active_connections[job_id])
            else:
                connections_to_send = []

        # Send to all currently connected clients
        disconnected: list[WebSocket] = []

        for connection in connections_to_send:
            try:
                await connection.send_json(progress_data)
            except Exception as e:
                logger.warning(f"Failed to send progress to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for connection in disconnected:
                    if (
                        job_id in self.active_connections
                        and connection in self.active_connections[job_id]
                    ):
                        self.active_connections[job_id].remove(connection)

    async def send_complete(
        self, job_id: str, success: bool, message: str, data: dict[str, Any] | None = None
    ) -> None:
        """
        Send completion message to all clients subscribed to a job.
        Stores in current state and schedules cleanup.
        """
        logger.info(
            f"send_complete() called for job {job_id}: success={success}, message={message[:50]}"
        )

        complete_data = {
            "type": "complete",
            "job_id": job_id,
            "success": success,
            "message": message,
            "data": data or {},
        }

        # Store state and snapshot connections atomically
        async with self._lock:
            self.current_state[job_id] = complete_data

            if job_id in self.active_connections:
                conn_count = len(self.active_connections[job_id])
                logger.info(
                    f"Sending completion to {conn_count} "
                    f"connected clients"
                )
                connections_to_send = list(self.active_connections[job_id])
            else:
                logger.info(
                    f"No active connections for job {job_id}, completion stored in state only"
                )
                connections_to_send = []

        for connection in connections_to_send:
            try:
                await connection.send_json(complete_data)
                logger.info("Successfully sent completion message to client")
            except Exception as e:
                logger.warning(f"Failed to send completion to client: {e}")
                async with self._lock:
                    if (
                        job_id in self.active_connections
                        and connection in self.active_connections[job_id]
                    ):
                        self.active_connections[job_id].remove(connection)

        # Clean up state after 60s (client has long since received it)
        self._schedule_cleanup(self._cleanup_state(job_id, delay=60))

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

        # Store state and snapshot connections atomically
        async with self._lock:
            self.current_state[job_id] = error_data

            if job_id in self.active_connections:
                logger.info(
                    f"Sending error to {len(self.active_connections[job_id])} connected clients"
                )
                connections_to_send = list(self.active_connections[job_id])
            else:
                logger.info(f"No active connections for job {job_id}, error stored in state only")
                connections_to_send = []

        for connection in connections_to_send:
            try:
                await connection.send_json(error_data)
                logger.info("Successfully sent error message to client")
            except Exception as e:
                logger.warning(f"Failed to send error to client: {e}")
                async with self._lock:
                    if (
                        job_id in self.active_connections
                        and connection in self.active_connections[job_id]
                    ):
                        self.active_connections[job_id].remove(connection)

        # Clean up state after 60s
        self._schedule_cleanup(self._cleanup_state(job_id, delay=60))

    async def send_cancelled(
        self, job_id: str, message: str = "Run cancelled"
    ) -> None:
        """Broadcast a 'cancelled' terminal message to all subscribers.

        Mirrors send_complete / send_error so the frontend can treat
        cancellation as a first-class terminal state.
        """
        logger.info(f"send_cancelled() called for job {job_id}")

        cancelled_data = {
            "type": "cancelled",
            "job_id": job_id,
            "message": message,
        }

        async with self._lock:
            self.current_state[job_id] = cancelled_data
            if job_id in self.active_connections:
                connections_to_send = list(self.active_connections[job_id])
            else:
                connections_to_send = []

        for connection in connections_to_send:
            try:
                await connection.send_json(cancelled_data)
            except Exception as e:
                logger.warning(f"Failed to send cancellation to client: {e}")
                async with self._lock:
                    if (
                        job_id in self.active_connections
                        and connection in self.active_connections[job_id]
                    ):
                        self.active_connections[job_id].remove(connection)

        self._schedule_cleanup(self._cleanup_state(job_id, delay=60))

    def get_connection_count(self, job_id: str) -> int:
        """Get number of active connections for a job."""
        return len(self.active_connections.get(job_id, []))

    def _schedule_cleanup(self, coro) -> None:
        """Schedule a cleanup coroutine and track the task for cancellation."""
        task = asyncio.create_task(coro)
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def close(self) -> None:
        """Cancel all pending cleanup tasks. Call on shutdown."""
        for task in self._cleanup_tasks:
            task.cancel()
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
        self._cleanup_tasks.clear()

    async def _cleanup_state(self, job_id: str, delay: int = 60) -> None:
        """Clean up current state for a job after a delay."""
        await asyncio.sleep(delay)
        if job_id in self.current_state:
            del self.current_state[job_id]
            logger.debug(f"Cleaned up state for job {job_id}")

    async def _cleanup_pending_start(
        self, task_id: str, delay: int = _PENDING_START_TIMEOUT
    ) -> None:
        """Remove a pending start if the frontend never sent 'ready'.

        Also settles the job row. Dropping only the in-memory callback left
        the row ``pending`` for ever: nothing can start it once the callback
        is gone, and the startup reconciliation only ever looked at
        ``running`` jobs, so an orphan survived every restart.
        """
        await asyncio.sleep(delay)
        removed = self._pending_starts.pop(task_id, None)
        if removed is not None:
            logger.warning(
                f"Cleaned up orphaned pending start for "
                f"task {task_id} (no 'ready' after {delay}s)"
            )
            _fail_orphaned_job(
                task_id,
                f"The app never connected to this job's progress channel "
                f"within {delay} seconds, so it was never started. "
                f"Run it again.",
            )


def _fail_orphaned_job(task_id: str, message: str) -> None:
    """Mark a job row failed, if ``task_id`` names one.

    Best-effort and quiet about misses: ``register_start`` is also used for
    model preparation, where the task id is a model id and no job row
    exists. Only a job still sitting in ``pending`` is touched, so this can
    never overwrite the outcome of work that did run.

    Imports are local to keep this module free of DB imports at import
    time; it is loaded by the API layer very early.
    """
    try:
        from app.db.base import get_session_factory
        from app.models import Job

        with get_session_factory()() as db:
            job = db.get(Job, task_id)
            if job is None or job.status != "pending":
                return
            job.status = "failed"
            job.error = message
            db.commit()
            logger.warning(f"Marked never-started job {task_id} as failed")
    except Exception as e:
        logger.error(f"Could not fail orphaned job {task_id}: {e}", exc_info=True)


# Global connection manager instance
ws_manager = ConnectionManager()
