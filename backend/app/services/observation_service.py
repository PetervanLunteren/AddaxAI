"""
Observations service — subprocess dispatcher for the Observations verify tab.

Delegates sort (greedy nearest-neighbor chain) and search (FAISS k-NN) to
ml/inference/similarity_script.py running in the addaxai-base conda
environment. The main backend process never imports numpy or faiss.

The subprocess emits NDJSON events on stdout (progress, result, error).
This module exposes:

- `stream_observations_subprocess(...)` — generator yielding raw NDJSON
  bytes lines for the streaming router endpoints to relay verbatim.
- `sort_detections` / `search_similar` — non-streaming convenience
  wrappers that drain the stream and return only the final result.
  Used by tests and any caller that doesn't care about progress.

The subprocess script is named for the underlying algorithm (cosine
similarity); this service is named for the feature it serves (the
Observations tab).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.api.schemas.observation import (
    ObservationFilters,
    SearchRequest,
    SearchResponse,
    SortRequest,
    SortResponse,
)
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.models import Project

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


def _filters_to_dict(filters: ObservationFilters) -> dict[str, Any]:
    """Convert Pydantic ObservationFilters to a JSON-safe dict."""
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


def stream_observations_subprocess(
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

    logger.info(f"Streaming observations subprocess: {operation} for project {project_id}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

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
        msg = f"Observations subprocess timed out after {timeout_s}s"
        logger.error(msg)
        yield (json.dumps({"type": "error", "message": msg}) + "\n").encode("utf-8")
        return
    finally:
        if process.stderr is not None:
            stderr = process.stderr.read()
            if stderr:
                for line in stderr.strip().splitlines():
                    logger.debug(f"similarity_script: {line}")

    if process.returncode != 0 and not saw_terminal_event:
        msg = f"Observations subprocess failed (exit {process.returncode})"
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
        raise RuntimeError("Observations subprocess produced no result event")
    return last_result


def _apply_project_threshold(
    filters: ObservationFilters, project_id: str, db: Session
) -> ObservationFilters:
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
    return stream_observations_subprocess("sort", project_id, params)


def stream_search(
    project_id: str, body: SearchRequest, db: Session
) -> Iterator[bytes]:
    """NDJSON event stream for the search endpoint."""
    params = _build_search_params(project_id, body, db)
    return stream_observations_subprocess("search", project_id, params)


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
