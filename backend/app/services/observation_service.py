"""
Observations service — subprocess dispatcher for the Observations verify tab.

Delegates sort (greedy nearest-neighbor chain) and search (FAISS k-NN) to
ml/inference/similarity_script.py running in the addaxai-base conda
environment. The main backend process never imports numpy or faiss.

The subprocess script is named for the underlying algorithm (cosine
similarity); this service is named for the feature it serves (the
Observations tab). Stats queries and detection summary building stay
in-process (pure SQL).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.api.schemas.observation import (
    DetectionSummary,
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


def _run_observations_subprocess(
    operation: str, project_id: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Run similarity_script.py as subprocess and return parsed JSON."""
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

    logger.info(f"Running observations subprocess: {operation} for project {project_id}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stdout, stderr = process.communicate(timeout=120)

    if stderr:
        for line in stderr.strip().splitlines():
            logger.debug(f"similarity_script: {line}")

    if process.returncode != 0:
        error_msg = stderr.strip() if stderr else "Unknown error"
        # Check for known user-facing errors
        if "Too many detections" in error_msg:
            raise ValueError(error_msg.replace("ERROR: ", ""))
        if "No embedding found" in error_msg:
            raise ValueError(error_msg.replace("ERROR: ", ""))
        raise RuntimeError(
            f"Observations computation failed: {error_msg}"
        )

    if not stdout.strip():
        raise RuntimeError("Observations script produced no output")

    return json.loads(stdout)


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


def sort_detections(
    project_id: str, body: SortRequest, db: Session
) -> SortResponse:
    """Sort detections by visual similarity via subprocess."""
    filters = _apply_project_threshold(body.filters, project_id, db)
    params = {
        "filters": _filters_to_dict(filters),
        "reverse": body.reverse,
    }
    result = _run_observations_subprocess("sort", project_id, params)
    return SortResponse(**result)


def search_similar(
    project_id: str, body: SearchRequest, db: Session
) -> SearchResponse:
    """Search for similar detections via subprocess."""
    filters = _apply_project_threshold(body.filters, project_id, db)
    params = {
        "filters": _filters_to_dict(filters),
        "anchor_detection_id": body.anchor_detection_id,
        "limit": body.limit,
        "threshold": body.threshold,
    }
    result = _run_observations_subprocess("search", project_id, params)
    return SearchResponse(**result)


def build_detection_summary(
    detection_id: str,
    meta: dict[str, Any],
    distance_to_centroid: float | None = None,
    similarity: float | None = None,
) -> DetectionSummary:
    """Build a DetectionSummary from metadata dict."""
    return DetectionSummary(
        detection_id=detection_id,
        file_id=meta["file_id"],
        label=meta["label"],
        label_confidence=meta["label_confidence"],
        confidence=meta["confidence"],
        category=meta["category"],
        verified=meta["verified"],
        classification_method=meta["classification_method"],
        distance_to_centroid=distance_to_centroid,
        similarity=similarity,
        site_name=meta.get("site_name"),
        deployment_id=meta.get("deployment_id"),
        timestamp=meta.get("timestamp"),
        crop_url=f"/api/detections/{detection_id}/crop?size=200",
    )
