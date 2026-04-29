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
