"""
First-run setup endpoints.

The desktop app is gated by a one-time setup wizard on first launch:
default model weights ship with the DMG (copied to ~/AddaxAI/models/ by
the lifespan hook), but env-addaxai-base must be downloaded once. The
wizard polls /api/setup/status, calls POST /api/setup/install-env to
start the install, and watches the progress fields update.

Polling rather than WebSocket: this is a one-shot UI staring at a
single progress bar for 5-15 minutes, polling at 1.5s costs nothing
and keeps the implementation small.

Module-level state intentionally resets when the server restarts: a
restart while installing is treated as "no install in progress", and
the user clicks Install again. The env_manager itself is idempotent
and resumes safely.
"""

import asyncio
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)
router = APIRouter(prefix="/api/setup", tags=["Setup"])

# Default models we expect to be installed for the basic MD+DINOv2-B
# workflow. Mirrors _DEFAULT_MODELS in services/bundled_models.py.
_REQUIRED_MODELS: tuple[tuple[str, str, str], ...] = (
    ("det", "MD5A-0-0", "md_v5a.0.0.pt"),
    ("emb", "DINOV2-VITB14", "dinov2_vitb14_pretrain.pth"),
)

_REQUIRED_ENV = "addaxai-base"


class _InstallState:
    """In-memory state for the env-install background task."""

    def __init__(self) -> None:
        self.in_progress: bool = False
        self.progress_pct: float = 0.0
        self.message: str = ""
        self.error: str | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Return True if started, False if already running."""
        with self._lock:
            if self.in_progress:
                return False
            self.in_progress = True
            self.progress_pct = 0.0
            self.message = "Starting install..."
            self.error = None
            return True

    def update(self, message: str, progress: float) -> None:
        with self._lock:
            self.message = message
            self.progress_pct = max(0.0, min(1.0, progress)) * 100.0

    def finish(self, error: str | None = None) -> None:
        with self._lock:
            self.in_progress = False
            self.progress_pct = 100.0 if error is None else self.progress_pct
            self.error = error
            self.message = "Install complete" if error is None else self.message


_install_state = _InstallState()
_env_manager: EnvironmentManager | None = None


def _get_env_manager() -> EnvironmentManager:
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager


def _models_present(models_dir: Path) -> bool:
    for category, model_id, fname in _REQUIRED_MODELS:
        if not (models_dir / category / model_id / fname).is_file():
            return False
    return True


def _env_present() -> bool:
    em = _get_env_manager()
    env_path = em.envs_dir / f"env-{_REQUIRED_ENV}"
    if not env_path.exists():
        return False
    # Reuse env_manager's own validation so a half-installed env counts
    # as missing and the wizard re-runs cleanly.
    return em._validate_env(env_path)


class SetupStatus(BaseModel):
    """Status surfaced to the wizard. Polled at ~1.5s."""

    ready: bool
    models_installed: bool
    env_installed: bool
    install_in_progress: bool
    progress_pct: float
    message: str
    error: str | None
    user_data_dir: str


@router.get("/status", response_model=SetupStatus)
def get_setup_status() -> SetupStatus:
    settings = get_settings()
    models_dir = settings.user_data_dir / "models"

    models_ok = _models_present(models_dir)
    env_ok = _env_present()

    return SetupStatus(
        ready=models_ok and env_ok,
        models_installed=models_ok,
        env_installed=env_ok,
        install_in_progress=_install_state.in_progress,
        progress_pct=_install_state.progress_pct,
        message=_install_state.message,
        error=_install_state.error,
        user_data_dir=str(settings.user_data_dir),
    )


def _build_stub_manifest() -> ModelManifest:
    """
    Build the minimal ModelManifest the env_manager needs to resolve the
    env yaml. Only `.env` is actually read by the install path; everything
    else is required by the schema but ignored at this site.
    """
    return ModelManifest(
        model_id="setup-stub",
        friendly_name="Default environment",
        emoji="⚙️",
        env=_REQUIRED_ENV,
        model_fname="setup-stub",
        description="Synthetic manifest used by the first-run setup wizard.",
        developer="AddaxAI",
        info_url="https://github.com/PetervanLunteren/AddaxAI-WebUI",
        min_app_version="0.1.0",
    )


def _install_env_blocking() -> None:
    """Sync worker that drives env_manager. Runs in a thread."""
    try:
        em = _get_env_manager()
        manifest = _build_stub_manifest()

        def progress_cb(message: str, progress: float) -> None:
            _install_state.update(message, progress)

        em.get_or_create_env(manifest, progress_cb)
        _install_state.finish(error=None)
    except Exception as e:
        logger.error(f"Setup env install failed: {e}", exc_info=True)
        _install_state.finish(error=str(e))


@router.post("/install-env", status_code=status.HTTP_202_ACCEPTED)
async def install_env() -> dict[str, str]:
    """
    Start env-addaxai-base creation in a background thread. Returns 202
    immediately; progress is polled via /status. 409 if already running.
    """
    if _env_present():
        return {"status": "already_installed"}

    if not _install_state.start():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Install already in progress",
        )

    # Run blocking install in a thread so the event loop stays responsive.
    asyncio.create_task(asyncio.to_thread(_install_env_blocking))
    return {"status": "started"}


# ---------------------------------------------------------------------------
# Reset application
#
# Wipes user data so the next launch starts from scratch (setup wizard runs
# again, models redeploy from bundle, env reinstalls). DB is preserved by
# default. The DB option uses a marker file consumed by the lifespan on the
# next launch so we don't have to fight SQLAlchemy's open connections here.
# ---------------------------------------------------------------------------

# Items wiped inline on POST /reset. None of these conflict with the
# running backend process: even if a worker is mid-write into a log or
# env, we're about to shut down anyway.
_WIPE_DIRS = ("logs", "envs", "models", "bin", "thumbnails", "crash-dumps")
_WIPE_FILES = (".last-shutdown-clean", ".last-launch-status.json")

# Read by lifespan() before init_db() on the next launch.
DB_WIPE_MARKER = ".wipe-db-on-next-launch"


class ResetRequest(BaseModel):
    """Body for POST /api/setup/reset."""

    confirmation: str
    wipe_database: bool = False


class ResetResponse(BaseModel):
    """Summary of what got wiped, for the caller to log/display."""

    removed_dirs: list[str]
    removed_files: list[str]
    db_wipe_scheduled: bool


def _safe_rmtree(path: Path) -> bool:
    """Remove path if it exists. Returns True if anything was removed."""
    if not path.exists():
        return False
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except Exception as e:
        logger.error(f"Failed to remove {path}: {e}", exc_info=True)
        return False


@router.post("/reset", response_model=ResetResponse)
def reset_application(req: ResetRequest) -> ResetResponse:
    """
    Wipe user data. Required confirmation string is the literal word
    "RESET" (case-sensitive) to avoid the dialog accidentally firing.

    Caller is expected to close the Electron app immediately afterwards
    so the next launch starts from a clean state.
    """
    if req.confirmation != "RESET":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation phrase did not match.",
        )

    settings = get_settings()
    user_data_dir = settings.user_data_dir

    removed_dirs: list[str] = []
    for name in _WIPE_DIRS:
        if _safe_rmtree(user_data_dir / name):
            removed_dirs.append(name)

    removed_files: list[str] = []
    for name in _WIPE_FILES:
        if _safe_rmtree(user_data_dir / name):
            removed_files.append(name)

    db_wipe_scheduled = False
    if req.wipe_database:
        # Drop a marker file. The lifespan reads this before init_db()
        # on the next launch and removes the SQLite files there, where
        # no SQLAlchemy connection is open yet.
        marker = user_data_dir / DB_WIPE_MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("scheduled")
        db_wipe_scheduled = True
        logger.warning(
            "DB wipe scheduled via marker file. The next launch will "
            "delete addaxai.db before initializing the database."
        )

    logger.warning(
        f"Application reset: removed dirs={removed_dirs} files={removed_files} "
        f"db_wipe_scheduled={db_wipe_scheduled}"
    )

    return ResetResponse(
        removed_dirs=removed_dirs,
        removed_files=removed_files,
        db_wipe_scheduled=db_wipe_scheduled,
    )
