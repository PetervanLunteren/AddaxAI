"""
Logging API endpoints.

Two responsibilities:
1. POST /api/logs - frontend forwards batched browser logs into backend.log
2. GET /api/logs/diagnostic-zip - bundles everything we'd need to debug a
   user issue (logs, system info, env state, db state, recent jobs) into
   a single ZIP the user emails to support. No upload, no telemetry.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash on unexpected errors (let FastAPI handle them)
"""

import io
import json
import platform
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/logs", tags=["Logging"])


class FrontendLogEntry(BaseModel):
    """Frontend log entry schema."""

    timestamp: str
    level: str  # "info" | "warn" | "error"
    message: str
    context: dict[str, object] | None = None


class FrontendLogsRequest(BaseModel):
    """Request body for forwarding frontend logs."""

    logs: list[FrontendLogEntry]


@router.post("", status_code=status.HTTP_201_CREATED)
def forward_frontend_logs(request: FrontendLogsRequest) -> dict[str, str]:
    """
    Forward frontend logs to backend log file.

    Receives batched logs from frontend and writes them to backend.log
    with [FRONTEND] prefix for easy filtering.

    Returns:
        Success message with count of logs received
    """
    frontend_logger = get_logger("frontend")

    for entry in request.logs:
        # Format: [FRONTEND] message {context}
        context_str = f" {entry.context}" if entry.context else ""
        log_message = f"[{entry.timestamp}] {entry.message}{context_str}"

        # Map frontend log levels to Python logging levels
        if entry.level == "error":
            frontend_logger.error(log_message)
        elif entry.level == "warn":
            frontend_logger.warning(log_message)
        else:  # info or anything else
            frontend_logger.info(log_message)

    logger.info(f"Received {len(request.logs)} frontend log entries")

    return {
        "status": "success",
        "message": f"Logged {len(request.logs)} entries",
    }


# ---------------------------------------------------------------------------
# Diagnostic ZIP
#
# Single-shot bundle the user emails to support when something goes wrong.
# Built in-memory and streamed: avoids leaving a temp file on disk and
# scales to a few hundred MB without trouble. We never upload it anywhere
# ourselves; transparency is the whole point.
# ---------------------------------------------------------------------------


def _collect_system_info() -> dict[str, object]:
    """OS / arch / Python / RAM / disk-free. No PII."""
    settings = get_settings()
    info: dict[str, object] = {
        "app_version": "0.1.0",  # bumped at release; see backend/app/main.py
        "environment": settings.environment,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "frozen": getattr(sys, "frozen", False),
        },
        "user_data_dir": str(settings.user_data_dir),
        "collected_at_utc": datetime.now(UTC).isoformat(),
    }
    # Disk free where the user data lives. Not all platforms expose this
    # via os.statvfs (Windows in particular), so be defensive.
    try:
        import shutil as _shutil

        usage = _shutil.disk_usage(settings.user_data_dir)
        info["disk_usage_bytes"] = {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
        }
    except Exception as e:
        info["disk_usage_error"] = str(e)
    return info


def _collect_env_status() -> dict[str, object]:
    """Which conda envs exist on disk and which validate."""
    try:
        from app.ml.environment_manager import EnvironmentManager

        em = EnvironmentManager()
        envs: list[dict[str, object]] = []
        if em.envs_dir.exists():
            for env_path in sorted(em.envs_dir.iterdir()):
                if not env_path.is_dir():
                    continue
                envs.append(
                    {
                        "name": env_path.name,
                        "path": str(env_path),
                        "valid": em._validate_env(env_path),
                    }
                )
        return {"envs_dir": str(em.envs_dir), "envs": envs}
    except Exception as e:
        return {"error": str(e)}


def _collect_models_inventory(user_data_dir: Path) -> dict[str, object]:
    """List of installed model dirs by category."""
    models_dir = user_data_dir / "models"
    inventory: dict[str, object] = {"models_dir": str(models_dir), "models": {}}
    if not models_dir.exists():
        inventory["note"] = "models_dir does not exist"
        return inventory
    for category_dir in sorted(models_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        items = []
        for model_dir in sorted(category_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            manifest = model_dir / "manifest.json"
            entry: dict[str, object] = {"id": model_dir.name}
            if manifest.is_file():
                try:
                    entry["manifest"] = json.loads(manifest.read_text())
                except Exception as e:
                    entry["manifest_error"] = str(e)
            # Total size of all files in the model dir.
            try:
                size = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
                entry["size_bytes"] = size
            except Exception:
                pass
            items.append(entry)
        inventory["models"][category_dir.name] = items  # type: ignore[index]
    return inventory


def _collect_db_info() -> dict[str, object]:
    """Alembic head and per-table row counts. No row contents."""
    info: dict[str, object] = {}
    try:
        from app.db.base import get_session_factory

        session_factory = get_session_factory()
        db = session_factory()
        try:
            head = db.execute(text("SELECT version_num FROM alembic_version")).scalar()
            info["alembic_version"] = head

            tables: dict[str, object] = {}
            table_names = db.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name"
                )
            ).scalars().all()
            for name in table_names:
                try:
                    count = db.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
                    tables[name] = count
                except SQLAlchemyError as e:
                    tables[name] = f"error: {e}"
            info["tables"] = tables
        finally:
            db.close()
    except Exception as e:
        info["error"] = str(e)
    return info


def _collect_recent_jobs(limit: int = 50) -> list[dict[str, object]]:
    """Last N jobs with their status + error fields. No payloads."""
    try:
        from app.db.base import get_session_factory

        session_factory = get_session_factory()
        db = session_factory()
        try:
            rows = db.execute(
                text(
                    "SELECT id, type, status, progress_current, progress_total, "
                    "error, created_at_utc, started_at_utc, completed_at_utc "
                    "FROM jobs ORDER BY created_at_utc DESC LIMIT :limit"
                ),
                {"limit": limit},
            ).mappings().all()
            return [dict(r) for r in rows]
        finally:
            db.close()
    except Exception as e:
        return [{"error": str(e)}]


def _build_diagnostic_zip() -> bytes:
    """
    Build the diagnostic bundle in-memory and return its bytes.

    Anything that fails inside one collector should not break the whole
    bundle: missing log file, dead DB, broken env_manager — each is
    captured as a *.error.txt entry in the ZIP so support can see what
    went wrong.
    """
    settings = get_settings()
    logs_dir = settings.user_data_dir / "logs"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # README explains the bundle contents to whoever opens it.
        zf.writestr(
            "README.txt",
            (
                "AddaxAI diagnostic report\n"
                "=========================\n\n"
                "This bundle contains:\n"
                "  - logs/                 application log files (rotating, last ~7 days)\n"
                "  - system.json           OS, Python, disk usage\n"
                "  - env-status.json       installed conda envs and their validity\n"
                "  - models.json           installed model directories and manifests\n"
                "  - db-info.json          alembic schema version and per-table row counts\n"
                "  - recent-jobs.json      last 50 jobs with status + error fields\n"
                "  - crash-sentinel.json   whether previous app shutdown was clean\n\n"
                "PII note: log files may contain absolute paths from the user's\n"
                "filesystem (e.g. paths to camera-trap folders). No detection data,\n"
                "image bytes, or DB row contents are included.\n\n"
                f"Collected at {datetime.now(UTC).isoformat()}\n"
            ),
        )

        # Logs directory: only include files that look like log files.
        # The dir occasionally accumulates stray non-log files from past
        # experiments (e.g. zero-byte .db leftovers); we don't want those
        # in a support bundle. Match on suffix instead of "everything".
        if logs_dir.is_dir():
            for src in sorted(logs_dir.rglob("*")):
                if not src.is_file():
                    continue
                # Accept .log, .log.1, .log.2, ... and .txt for any
                # ad-hoc log files a future layer might create.
                name = src.name
                is_log = (
                    name.endswith(".log")
                    or ".log." in name  # rotated: backend.log.1
                    or name.endswith(".txt")
                )
                if not is_log:
                    continue
                try:
                    zf.write(src, arcname=f"logs/{src.relative_to(logs_dir)}")
                except Exception as e:
                    zf.writestr(
                        f"logs/{src.name}.error.txt",
                        f"failed to add {src}: {e}\n",
                    )
        else:
            zf.writestr("logs/.missing.txt", f"no logs directory at {logs_dir}\n")

        for name, collector in (
            ("system.json", _collect_system_info),
            ("env-status.json", _collect_env_status),
            ("db-info.json", _collect_db_info),
            ("recent-jobs.json", _collect_recent_jobs),
        ):
            try:
                payload = collector()
                zf.writestr(name, json.dumps(payload, indent=2, default=str))
            except Exception as e:
                zf.writestr(f"{name}.error.txt", f"collector raised: {e}\n")

        # Models inventory needs the user_data_dir; called separately
        # because the signature differs.
        try:
            zf.writestr(
                "models.json",
                json.dumps(
                    _collect_models_inventory(settings.user_data_dir),
                    indent=2,
                    default=str,
                ),
            )
        except Exception as e:
            zf.writestr("models.json.error.txt", f"collector raised: {e}\n")

        # Crash sentinel snapshot. Electron writes this on each launch
        # capturing whether the previous shutdown was clean (see
        # snapshotPreviousShutdown in electron/src/main.ts). If the file
        # is missing, the user is either on a non-Electron run (uvicorn
        # only) or has never launched the desktop app yet.
        snapshot = settings.user_data_dir / ".last-launch-status.json"
        if snapshot.is_file():
            try:
                zf.writestr("crash-sentinel.json", snapshot.read_text())
            except Exception as e:
                zf.writestr("crash-sentinel.json", json.dumps({"error": str(e)}))
        else:
            zf.writestr(
                "crash-sentinel.json",
                json.dumps({"note": "no launch-status snapshot present"}),
            )

    return buf.getvalue()


@router.get("/last-launch-status")
def last_launch_status() -> dict[str, object]:
    """
    Return whether the previous AddaxAI shutdown was clean. The frontend
    polls this once on first load and shows a "previous run crashed"
    banner if not.

    The snapshot is written by Electron at every launch. When the app is
    running under uvicorn directly (dev), the snapshot may not exist; we
    return previous_shutdown_clean=true in that case so dev never gets
    spurious banners.
    """
    settings = get_settings()
    snapshot = settings.user_data_dir / ".last-launch-status.json"
    if not snapshot.is_file():
        return {"previous_shutdown_clean": True, "snapshot_present": False}
    try:
        data = json.loads(snapshot.read_text())
        return {
            "previous_shutdown_clean": bool(
                data.get("previous_shutdown_clean", True)
            ),
            "snapshot_present": True,
            "current_launch_at": data.get("current_launch_at"),
        }
    except Exception as e:
        logger.warning(f"Could not parse last-launch-status snapshot: {e}")
        return {"previous_shutdown_clean": True, "snapshot_present": False}


@router.get("/diagnostic-zip")
def diagnostic_zip() -> StreamingResponse:
    """
    Build and return a ZIP bundle the user can email to support.

    Synchronous because this is a once-per-bug-report operation and
    the I/O cost is dominated by reading log files which are local SSD.
    """
    logger.info("Diagnostic ZIP requested")
    payload = _build_diagnostic_zip()
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    filename = f"addaxai-diagnostics-{timestamp}.zip"
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
