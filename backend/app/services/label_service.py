"""
Labels service — subprocess dispatcher for the Labels verify tab.

Delegates sort (greedy nearest-neighbor chain), search (FAISS k-NN), and
cohort grouping (descendant-promotion review panel) to
ml/inference/similarity_script.py running in the addaxai-base conda
environment. The main backend process never imports numpy or faiss.

The subprocess emits NDJSON events on stdout (progress, result, error).
This module exposes:

- `stream_labels_subprocess(...)` — generator yielding raw NDJSON
  bytes lines for the streaming router endpoints to relay verbatim.
- `sort_detections` / `search_similar` — non-streaming convenience
  wrappers that drain the stream and return only the final result.
  Used by tests and any caller that doesn't care about progress.

The subprocess script is named for the underlying algorithm (cosine
similarity); this service is named for the feature it serves (the
Labels tab).
"""

from __future__ import annotations

import asyncio
import json
import queue
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from app.api.schemas.label import (
    LabelFilters,
    SearchRequest,
    SearchResponse,
    SortRequest,
    SortResponse,
)
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.models import Project
from app.utils.subprocess_env import clean_python_env

logger = get_logger(__name__)

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "ml" / "inference" / "similarity_script.py"


def _get_env_python() -> Path:
    """Get Python path from the addaxai-base conda environment."""
    env_manager = EnvironmentManager()
    try:
        return env_manager.get_python("env-addaxai-base")
    except FileNotFoundError:
        raise FileNotFoundError(
            "ML environment not found. "
            "Run an analysis with a detection model first to set up the ML environment."
        ) from None


def _get_db_path() -> str:
    """Extract file path from database URL (strips sqlite:/// prefix)."""
    url = get_settings().database_url
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    raise ValueError(f"Unsupported database URL format: {url}")


def _filters_to_dict(filters: LabelFilters) -> dict[str, Any]:
    """Convert Pydantic LabelFilters to a JSON-safe dict."""
    d: dict[str, Any] = {}
    if filters.labels:
        # Strip :unspecified suffix from rolled-up taxonomy leaf IDs
        d["labels"] = [
            s.removesuffix(":unspecified") for s in filters.labels
        ]
    if filters.site_ids:
        d["site_ids"] = filters.site_ids
    if filters.date_from is not None:
        d["date_from"] = filters.date_from.isoformat()
    if filters.date_to is not None:
        d["date_to"] = filters.date_to.isoformat()
    if filters.min_confidence is not None:
        d["min_confidence"] = filters.min_confidence
    if filters.max_confidence is not None:
        d["max_confidence"] = filters.max_confidence
    if filters.min_label_confidence is not None:
        d["min_label_confidence"] = filters.min_label_confidence
    if filters.max_label_confidence is not None:
        d["max_label_confidence"] = filters.max_label_confidence
    if filters.project_floor is not None:
        d["project_floor"] = filters.project_floor
    if filters.category:
        d["category"] = filters.category
    if filters.verified is not None:
        d["verified"] = filters.verified
    return d


def stream_labels_subprocess(
    operation: str, project_id: str, params: dict[str, Any]
) -> Iterator[bytes]:
    """Run similarity_script.py and yield each NDJSON line from its stdout.

    Each yielded value already ends with `\\n` and is one of:

    - `{"type": "progress", "phase": "...", "done": N, "total": M}`
    - `{"type": "result", ...}`
    - `{"type": "error", "message": "..."}`

    The router relays these verbatim to the HTTP client. On non-zero
    subprocess exit without an explicit error event (rare; usually the
    subprocess emits one before exit), an error line is synthesised.
    """
    python_path = _get_env_python()
    db_path = _get_db_path()

    cmd = [
        str(python_path),
        str(_SCRIPT_PATH),
        "--db-path", db_path,
        "--project-id", project_id,
        "--operation", operation,
        "--params", json.dumps(params, default=str),
    ]

    logger.info(f"Streaming labels subprocess: {operation} for project {project_id}")

    env = clean_python_env(PYTHONUNBUFFERED="1")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    # 60s base + 6s per 1k cap. 20k → 180s, 50k → 360s. Subprocess
    # progress reaches us steadily so this is a hard ceiling, not the
    # typical wait. process.wait() at the end enforces it.
    cap = int(params.get("max_detections", 20000))
    timeout_s = 60 + (cap // 1000) * 6

    saw_terminal_event = False
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            # Sanity-check that the line is JSON before relaying. If it
            # isn't (subprocess crashed mid-write, env-addaxai-base
            # printed a warning), skip it rather than corrupt the
            # NDJSON stream.
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"similarity_script non-JSON stdout: {line!r}")
                continue
            if event.get("type") in ("result", "error"):
                saw_terminal_event = True
            yield (line + "\n").encode("utf-8")

        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        msg = f"Labels subprocess timed out after {timeout_s}s"
        logger.error(msg)
        yield (json.dumps({"type": "error", "message": msg}) + "\n").encode("utf-8")
        return
    except GeneratorExit:
        # The client disconnected mid-stream (browser navigated away,
        # tab closed, user hit refresh). FastAPI cancels the generator;
        # without an explicit kill the subprocess keeps running until
        # the timeout, holding one of the browser's per-host connection
        # slots and the cohort's compute. Kill it so subsequent
        # requests aren't queued behind a zombie.
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            logger.info(
                "Labels subprocess killed on client disconnect"
            )
        raise
    finally:
        if process.stderr is not None:
            stderr = process.stderr.read()
            if stderr:
                for line in stderr.strip().splitlines():
                    logger.debug(f"similarity_script: {line}")

    if process.returncode != 0 and not saw_terminal_event:
        msg = f"Labels subprocess failed (exit {process.returncode})"
        logger.error(msg)
        yield (json.dumps({"type": "error", "message": msg}) + "\n").encode("utf-8")


def _drain_to_result(events: Iterator[bytes]) -> dict[str, Any]:
    """Consume an event stream and return just the final `result` payload.

    Raises ValueError on `error` events. Used by callers that don't need
    progress (tests, plain-JSON wrappers).
    """
    last_result: dict[str, Any] | None = None
    for raw in events:
        event = json.loads(raw.decode("utf-8"))
        if event["type"] == "result":
            last_result = {k: v for k, v in event.items() if k != "type"}
        elif event["type"] == "error":
            raise ValueError(event["message"])
    if last_result is None:
        raise RuntimeError("Labels subprocess produced no result event")
    return last_result


def _apply_project_threshold(
    filters: LabelFilters, project_id: str, db: Session
) -> LabelFilters:
    """Inject the project's detection threshold as `project_floor`.

    The floor applies the `(confidence >= floor OR verified)` override
    rule shared with events / files. The user's `min_confidence` slider
    stays untouched and is applied LITERALLY by the subprocess so a
    verified low-confidence detection cannot bypass a narrow user range.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if project:
        filters = filters.model_copy(
            update={"project_floor": project.detection_threshold}
        )
    return filters


def _build_sort_params(
    project_id: str, body: SortRequest, db: Session
) -> dict[str, Any]:
    filters = _apply_project_threshold(body.filters, project_id, db)
    return {
        "filters": _filters_to_dict(filters),
        "sort": body.sort,
        "max_detections": body.max_detections,
    }


def _build_search_params(
    project_id: str, body: SearchRequest, db: Session
) -> dict[str, Any]:
    filters = _apply_project_threshold(body.filters, project_id, db)
    return {
        "filters": _filters_to_dict(filters),
        "anchor_detection_id": body.anchor_detection_id,
        "limit": body.limit,
        "threshold": body.threshold,
        "max_detections": body.max_detections,
    }


def stream_sort(
    project_id: str, body: SortRequest, db: Session
) -> Iterator[bytes]:
    """NDJSON event stream for the sort endpoint."""
    params = _build_sort_params(project_id, body, db)
    return stream_labels_subprocess("sort", project_id, params)


def stream_search(
    project_id: str, body: SearchRequest, db: Session
) -> Iterator[bytes]:
    """NDJSON event stream for the search endpoint."""
    params = _build_search_params(project_id, body, db)
    return stream_labels_subprocess("search", project_id, params)


def stream_cohorts(
    project_id: str, min_count: int, max_cohorts: int, db: Session
) -> Iterator[bytes]:
    """NDJSON event stream for the cohorts endpoint.

    Applies the project's `(confidence >= threshold OR verified)` floor
    so the pill counts the same population the suggestions grid will
    actually load. Without this, below-threshold detections form cohorts
    that the grid silently filters out, and the pill drifts above what
    "Review" can ever show.

    Still no user filters (sites / labels / dates): the pill is a global
    quality signal, independent of the filter bar.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    params: dict[str, Any] = {
        "min_count": min_count,
        "max_cohorts": max_cohorts,
        "filters": {
            "project_floor": project.detection_threshold if project else 0.0,
        },
    }
    return stream_labels_subprocess("cohorts", project_id, params)


def sort_detections(
    project_id: str, body: SortRequest, db: Session
) -> SortResponse:
    """Drain the sort stream into a single SortResponse.

    Used by tests; production traffic goes through `stream_sort` so the
    UI can render a progress bar.
    """
    return SortResponse(**_drain_to_result(stream_sort(project_id, body, db)))


def search_similar(
    project_id: str, body: SearchRequest, db: Session
) -> SearchResponse:
    """Drain the search stream into a single SearchResponse."""
    return SearchResponse(**_drain_to_result(stream_search(project_id, body, db)))


# ── Async streaming with client-disconnect handling ─────────────────────


async def stream_labels_subprocess_async(
    request: Request,
    operation: str,
    project_id: str,
    params: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Async variant that polls ``request.is_disconnected()`` and kills
    the subprocess when the client navigates away mid-stream.

    The sync version above can't react to client disconnect while it is
    blocked on ``process.stdout.readline()``: Starlette can only deliver
    a ``GeneratorExit`` at a yield point, and the subprocess holds the
    generator off the yield point indefinitely. In the wild that meant a
    refreshed tab left a similarity subprocess running until its 180 s
    timeout, holding one of the browser's six per-host connection slots
    so the next page load queued behind it. This generator reads stdout on
    a background thread and pops with a 0.5 s queue-poll tick, rechecking
    ``is_disconnected`` between ticks, so a refresh frees the slot within
    half a second. (Pure ``select`` on the pipe is not portable: Windows
    only selects on sockets, not pipes.)
    """
    python_path = _get_env_python()
    db_path = _get_db_path()

    cmd = [
        str(python_path),
        str(_SCRIPT_PATH),
        "--db-path", db_path,
        "--project-id", project_id,
        "--operation", operation,
        "--params", json.dumps(params, default=str),
    ]
    logger.info(
        f"Streaming labels subprocess (async): {operation} for project {project_id}"
    )
    env = clean_python_env(PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    cap = int(params.get("max_detections", 20000))
    timeout_s = 60 + (cap // 1000) * 6
    deadline = time.monotonic() + timeout_s
    loop = asyncio.get_running_loop()
    saw_terminal_event = False

    # Windows `select` only accepts sockets, not subprocess pipes, so the old
    # `select.select([process.stdout], ...)` raised WinError 10038 there and the
    # whole sort/cohort stream failed. A daemon reader thread does blocking line
    # reads into a queue instead (same approach as the sync variant above);
    # read_next() pops with a 0.5 s timeout, preserving the "" / line / None
    # contract and the re-check-disconnect-during-silence cadence, cross-platform.
    line_queue: queue.Queue[str | None] = queue.Queue()

    def pump_stdout() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)  # EOF sentinel

    threading.Thread(target=pump_stdout, daemon=True).start()

    def read_next() -> str | None:
        """One line, '' on a silent 0.5 s tick (caller re-checks disconnect),
        or None on EOF."""
        try:
            return line_queue.get(timeout=0.5)
        except queue.Empty:
            return ""

    try:
        while True:
            if await request.is_disconnected():
                logger.info(
                    f"Client disconnected; killing {operation} subprocess for {project_id}"
                )
                break
            if time.monotonic() > deadline:
                msg = f"Labels subprocess timed out after {timeout_s}s"
                logger.error(msg)
                yield (json.dumps({"type": "error", "message": msg}) + "\n").encode("utf-8")
                break

            chunk = await loop.run_in_executor(None, read_next)
            if chunk is None:
                break  # EOF
            if chunk == "":
                continue  # silent tick, re-check disconnect

            line = chunk.rstrip("\n")
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"similarity_script non-JSON stdout: {line!r}")
                continue
            if event.get("type") in ("result", "error"):
                saw_terminal_event = True
            yield (line + "\n").encode("utf-8")
    finally:
        if process.poll() is None:
            process.kill()
            try:
                await loop.run_in_executor(None, lambda: process.wait(timeout=2))
            except subprocess.TimeoutExpired:
                pass
        if process.stderr is not None:
            stderr = await loop.run_in_executor(None, process.stderr.read)
            if stderr:
                for line in stderr.strip().splitlines():
                    logger.debug(f"similarity_script: {line}")

    if process.returncode is not None and process.returncode != 0 and not saw_terminal_event:
        msg = f"Labels subprocess failed (exit {process.returncode})"
        logger.error(msg)
        yield (json.dumps({"type": "error", "message": msg}) + "\n").encode("utf-8")


async def stream_sort_async(
    request: Request, project_id: str, body: SortRequest, db: Session
) -> AsyncIterator[bytes]:
    params = _build_sort_params(project_id, body, db)
    async for chunk in stream_labels_subprocess_async(
        request, "sort", project_id, params
    ):
        yield chunk


async def stream_search_async(
    request: Request, project_id: str, body: SearchRequest, db: Session
) -> AsyncIterator[bytes]:
    params = _build_search_params(project_id, body, db)
    async for chunk in stream_labels_subprocess_async(
        request, "search", project_id, params
    ):
        yield chunk


async def stream_cohorts_async(
    request: Request,
    project_id: str,
    min_count: int,
    max_cohorts: int,
    db: Session,
) -> AsyncIterator[bytes]:
    project = db.query(Project).filter(Project.id == project_id).first()
    params: dict[str, Any] = {
        "min_count": min_count,
        "max_cohorts": max_cohorts,
        "filters": {
            "project_floor": project.detection_threshold if project else 0.0,
        },
    }
    async for chunk in stream_labels_subprocess_async(
        request, "cohorts", project_id, params
    ):
        yield chunk
