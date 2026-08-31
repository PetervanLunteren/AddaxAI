"""Cooperative cancellation for long-running ML jobs.

The worker runs an asyncio task that offloads subprocesses to an
executor. To cancel mid-phase we need to both (a) remember that a
cancel was requested and (b) kill the currently-running subprocess
tree immediately so the executor function unblocks.

Call sites:

    # Websocket handler (asyncio task):
    request_cancel(job_id)   # sets flag and kills current subprocess

    # ML module consuming proc.stdout (executor thread):
    with track_subprocess(job_id, proc):
        for line in proc.stdout:
            ...

    # After the subprocess returns, worker checks:
    if is_cancel_requested(job_id):
        raise JobCancelledError()

    # Worker, on any terminal outcome:
    clear_cancel(job_id)
"""

import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class JobCancelledError(Exception):
    """Raised when a subprocess exited because the job was cancelled."""


# Writes happen from executor threads (track_subprocess) and from the
# asyncio loop (request_cancel). A plain lock is enough; we only need
# atomic read-and-then-maybe-kill in request_cancel.
_lock = threading.Lock()
_current_process: dict[str, subprocess.Popen] = {}
_cancel_requested: set[str] = set()

# How long to wait for SIGTERM to land before escalating to SIGKILL
# (Unix only — Windows `taskkill /F` is immediate).
_SIGKILL_AFTER_SECONDS = 3.0


def request_cancel(job_id: str) -> None:
    """Flag the job as cancelled and kill its current subprocess tree."""
    with _lock:
        _cancel_requested.add(job_id)
        proc = _current_process.get(job_id)
    if proc is not None and proc.poll() is None:
        logger.info(f"Cancel: killing subprocess tree for job {job_id}, pid {proc.pid}")
        _kill_tree(proc)


def is_cancel_requested(job_id: str) -> bool:
    with _lock:
        return job_id in _cancel_requested


def clear_cancel(job_id: str) -> None:
    """Forget all cancel state for a job. Call on any terminal outcome."""
    with _lock:
        _cancel_requested.discard(job_id)
        _current_process.pop(job_id, None)


def kill_all_tracked() -> int:
    """Kill every tracked subprocess tree. Call once at backend shutdown.

    The ML subprocesses run in their own session (see `popen_group`), so
    stopping the backend does not stop them: a detector left behind by an
    app quit ran on for hours, then deleted its own checkpoint and wrote
    its output where nothing would ever read it. Returns how many trees
    were killed.
    """
    with _lock:
        tracked = list(_current_process.items())
    killed = 0
    for job_id, proc in tracked:
        if proc.poll() is None:
            logger.info(
                f"Shutdown: killing subprocess tree for job {job_id}, pid {proc.pid}"
            )
            _kill_tree(proc)
            killed += 1
    return killed


@contextmanager
def track_subprocess(job_id: str | None, proc: subprocess.Popen):
    """Register `proc` as the job's current subprocess for the duration
    of the `with` block. If a cancel was requested between spawning
    `proc` and entering this block, kill it immediately.

    `job_id` of None is a no-op (useful for callers that don't thread
    a job id, e.g. environment setup before a job exists).
    """
    if job_id is None:
        yield
        return

    with _lock:
        _current_process[job_id] = proc
        already_cancelled = job_id in _cancel_requested
    if already_cancelled:
        logger.info(
            f"Cancel: pre-existing flag for job {job_id}, "
            f"killing new subprocess pid {proc.pid}"
        )
        _kill_tree(proc)
    try:
        yield
    finally:
        with _lock:
            if _current_process.get(job_id) is proc:
                _current_process.pop(job_id, None)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the subprocess and every descendant it spawned.

    Unix: SIGTERM the process group, poll, SIGKILL if still alive.
    Windows: `taskkill /F /T /PID` takes the whole tree down in one shot.
    Tolerates processes that already exited.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + _SIGKILL_AFTER_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    logger.warning(f"Cancel: SIGTERM ignored by pid {proc.pid}, sending SIGKILL")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
