"""
First-run setup endpoints.

The desktop app is gated by a one-time setup wizard on first launch.
Setup has two parts: install env-addaxai-base via micromamba, and
download the default model weights (MDv5A + DINOv2-S) from HuggingFace.
Both run in a background thread driven by POST /api/setup/install-env;
the wizard polls /api/setup/status at ~1.5s and watches progress.

Polling rather than WebSocket: this is a one-shot UI staring at a
single progress bar for 10-30 minutes, polling at 1.5s costs nothing
and keeps the implementation small.

Module-level state intentionally resets when the server restarts: a
restart while installing is treated as "no install in progress", and
the user clicks Install again. Both the env manager and the HF
downloader are idempotent and resume safely.
"""

import asyncio
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.environment_manager import EnvironmentManager
from app.ml.model_storage import ModelStorage
from app.ml.schemas.model_manifest import ModelManifest
from app.services import legacy_install
from app.utils.fs_remove import safe_rmtree

logger = get_logger(__name__)
router = APIRouter(prefix="/api/setup", tags=["Setup"])

# Default models the wizard installs from HuggingFace. Held inline rather
# than read from the catalog so first-run setup works even if the catalog
# updater is still running or unreachable. The HF repos sit under the
# Addax-Data-Science org (same convention ModelStorage falls back to).
_DEFAULT_MODELS: tuple[dict, ...] = (
    {
        "type_dir": "det",
        "category": "detection",
        "model_id": "MD5A-0-0",
        "friendly_name": "MegaDetector v5A",
        "emoji": "🦌",
        "model_fname": "md_v5a.0.0.pt",
        "hf_repo": "Addax-Data-Science/MD5A-0-0",
    },
    {
        "type_dir": "emb",
        "category": "embedding",
        "model_id": "DINOV2-VITS14",
        "friendly_name": "DINOv2 ViT-S/14",
        "emoji": "🧠",
        "model_fname": "dinov2_vits14_pretrain.pth",
        "hf_repo": "Addax-Data-Science/DINOV2-VITS14",
    },
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
    for spec in _DEFAULT_MODELS:
        weight = models_dir / spec["type_dir"] / spec["model_id"] / spec["model_fname"]
        if not weight.is_file():
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


def _build_env_manifest(env_name: str = _REQUIRED_ENV) -> ModelManifest:
    """
    Build the minimal ModelManifest the env_manager needs to resolve the
    env yaml. Only `.env` is actually read by the install path; everything
    else is required by the schema but ignored at this site.

    Parameterised on env_name so the rebuild flow can drive a fresh
    install of any shipped env (`pytorch`, `tensorflow-v1`, etc.), not
    just the default one the wizard installs at first launch.
    """
    return ModelManifest(
        model_id=f"setup-stub-{env_name}",
        friendly_name=f"{env_name} environment",
        emoji="⚙️",
        env=env_name,
        model_fname="setup-stub",
        description="Synthetic manifest used by the first-run setup wizard.",
        developer="AddaxAI",
        info_url="https://github.com/PetervanLunteren/AddaxAI-WebUI",
        min_app_version="0.1.0",
    )


def _build_default_model_manifest(spec: dict) -> ModelManifest:
    """
    Build a synthetic ModelManifest pointing at the HF repo for one of
    the default models. ModelStorage uses model_category for the
    on-disk path layout and hf_repo for the source.
    """
    m = ModelManifest(
        model_id=spec["model_id"],
        friendly_name=spec["friendly_name"],
        emoji=spec["emoji"],
        env=_REQUIRED_ENV,
        model_fname=spec["model_fname"],
        hf_repo=spec["hf_repo"],
        description=f"Default {spec['category']} model installed by the setup wizard.",
        developer="AddaxAI",
        info_url="https://github.com/PetervanLunteren/AddaxAI-WebUI",
        min_app_version="0.1.0",
    )
    m.model_category = spec["category"]
    return m


def _install_env_blocking(force_envs: tuple[str, ...] = ()) -> None:
    """
    Sync worker driving setup: env install plus HF downloads of the
    default model weights. Runs in a thread. All steps are idempotent so
    retrying after a partial failure picks up where it left off.

    `force_envs` lets the drift-rebuild flow request that specific
    envs be wiped and rebuilt regardless of whether they already exist
    on disk. Caller passes env names like "addaxai-base" or "pytorch".
    """
    try:
        settings = get_settings()
        models_dir = settings.user_data_dir / "models"
        storage = ModelStorage(models_dir)
        em = _get_env_manager()

        # Wipe envs explicitly requested for rebuild before the
        # presence check below decides whether to add an install step.
        # Out-of-loop so the wipes are visible regardless of step
        # ordering.
        for env_name in force_envs:
            wipe_path = em.envs_dir / f"env-{env_name}"
            if wipe_path.exists():
                logger.info(
                    f"Force-rebuild: wiping existing env at {wipe_path}"
                )
                try:
                    em._safe_rmtree(wipe_path)
                except Exception as e:
                    logger.error(
                        f"Failed to wipe env {env_name} for rebuild: {e}",
                        exc_info=True,
                    )

        # Each step: (label_for_logs, fn(progress_cb)). Skipped if already
        # complete so retries go straight to whatever's missing.
        steps: list[tuple[str, Callable[[Callable[[str, float], None]], None]]] = []

        if not _env_present():
            def _env_step(cb: Callable[[str, float], None]) -> None:
                _get_env_manager().get_or_create_env(_build_env_manifest(), cb)
            steps.append(("Analysis environment", _env_step))

        # Add explicit rebuild steps for any forced env that isn't the
        # default one (the default is already covered above via
        # _env_present()). Order doesn't matter for correctness; users
        # see them sequentially in the progress modal.
        for env_name in force_envs:
            if env_name == _REQUIRED_ENV:
                continue
            def _force_env_step(
                cb: Callable[[str, float], None],
                _name: str = env_name,
            ) -> None:
                _get_env_manager().get_or_create_env(
                    _build_env_manifest(_name), cb
                )
            steps.append((f"Environment ({env_name})", _force_env_step))

        for spec in _DEFAULT_MODELS:
            weight = (
                models_dir / spec["type_dir"] / spec["model_id"] / spec["model_fname"]
            )
            if weight.is_file():
                continue
            manifest = _build_default_model_manifest(spec)

            def _model_step(
                cb: Callable[[str, float], None], _m: ModelManifest = manifest
            ) -> None:
                storage.download_weights(_m, cb)

            steps.append((spec["friendly_name"], _model_step))

        if not steps:
            _install_state.finish(error=None)
            return

        # Equal-slot progress allocation. Env install dominates real time
        # but the bar stays alive either way; weighting it would just
        # surprise users when the downloads "sprint" through 80-100%.
        # Prefix messages with Step N/M when there is more than one step
        # so the user knows which phase is running. Mirrors the format
        # used by the in-app model install (routers/ml_models.py).
        n = len(steps)
        for i, (label, run) in enumerate(steps):
            slot_start = i / n
            slot_span = 1.0 / n
            prefix = f"Step {i + 1}/{n} - " if n > 1 else ""
            logger.info(f"Setup step {i + 1}/{n}: {label}")

            def cb(
                msg: str,
                prog: float,
                _s: float = slot_start,
                _sp: float = slot_span,
                _p: str = prefix,
            ) -> None:
                _install_state.update(f"{_p}{msg}", _s + prog * _sp)

            run(cb)

        _install_state.finish(error=None)
    except Exception as e:
        logger.error(f"Setup install failed: {e}", exc_info=True)
        _install_state.finish(error=str(e))


class InstallEnvRequest(BaseModel):
    """
    Request body for POST /install-env.

    First-run setup posts an empty body. The drift-rebuild flow posts
    `force_envs` to wipe and recreate specific envs; useful when a
    bundled YAML moved on but the env on disk wasn't picked up.
    """

    force_envs: list[str] | None = None


@router.post("/install-env", status_code=status.HTTP_202_ACCEPTED)
async def install_env(
    request: InstallEnvRequest | None = None,
) -> dict[str, str]:
    """
    Start setup (env + default-model downloads) in a background thread.
    Returns 202 immediately; progress is polled via /status. 409 if
    already running. Idempotent: each step no-ops when complete.

    With `force_envs` set, the listed envs are wiped and rebuilt before
    the regular install loop runs, regardless of whether they already
    exist. Used by the drift-rebuild button.
    """
    settings = get_settings()
    force_envs = tuple(request.force_envs or ()) if request else ()

    # Skip-fast applies only when nothing is being forced. A force
    # request always triggers the install loop.
    if (
        not force_envs
        and _env_present()
        and _models_present(settings.user_data_dir / "models")
    ):
        return {"status": "already_installed"}

    # Disk-space pre-flight. Unpacked env (~3 GB) plus default model
    # weights (~2 GB) plus pip working space pushes the real footprint
    # past 7 GB. Failing fast with a clear message beats a cryptic
    # OSError two minutes into a download.
    _check_disk_space(settings.user_data_dir)

    if not _install_state.start():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Install already in progress",
        )

    # Run blocking install in a thread so the event loop stays responsive.
    asyncio.create_task(
        asyncio.to_thread(_install_env_blocking, force_envs)
    )
    return {"status": "started"}


# Required free space at the user-data drive before the install starts.
# Headroom for env (~3 GB) + default models (~2 GB) + pip working space.
_REQUIRED_FREE_BYTES = 7 * 1024**3


def _check_disk_space(user_data_dir: Path) -> None:
    """
    Raise HTTPException 507 (Insufficient Storage) if the volume backing
    the user data directory has less than _REQUIRED_FREE_BYTES free.

    Falls back silently if disk_usage isn't supported on the path (e.g.
    a network mount during edge-case Windows configurations); we'd
    rather attempt the install than block on a stat call that lies.
    """
    try:
        # If the dir doesn't exist yet, fall back to its first existing
        # ancestor — disk_usage needs a real path.
        target = user_data_dir
        while not target.exists():
            if target.parent == target:
                return
            target = target.parent
        free = shutil.disk_usage(target).free
    except OSError as e:
        logger.warning(
            f"Disk-space pre-flight could not stat {user_data_dir}: {e}; "
            f"proceeding without check"
        )
        return

    if free < _REQUIRED_FREE_BYTES:
        free_gb = free / 1024**3
        required_gb = _REQUIRED_FREE_BYTES / 1024**3
        raise HTTPException(
            status_code=507,
            detail=(
                f"Not enough free disk space at {user_data_dir}. "
                f"Setup needs about {required_gb:.0f} GB free; "
                f"only {free_gb:.1f} GB available."
            ),
        )


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
        if safe_rmtree(user_data_dir / name):
            removed_dirs.append(name)

    removed_files: list[str] = []
    for name in _WIPE_FILES:
        if safe_rmtree(user_data_dir / name):
            removed_files.append(name)

    # Drop the cached EnvironmentManager so the next setup attempt rebuilds
    # one against the wiped filesystem. Without this, the cached manager
    # still points at the now-deleted ~/AddaxAI/bin/micromamba and the next
    # install crashes with ENOENT inside subprocess. Production normally
    # quits + relaunches after reset, but dev mode (uvicorn in a browser
    # tab) keeps the process alive, so the cache must be invalidated here.
    global _env_manager
    _env_manager = None

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


# ---------------------------------------------------------------------------
# Legacy AddaxAI removal
#
# Offers to delete a legacy AddaxAI (v5 / v6) install so a machine that
# upgraded isn't left carrying two full copies. All the path knowledge lives
# in app/services/legacy_install.py; these endpoints are just the plumbing.
# ---------------------------------------------------------------------------

_RETRY_MESSAGE = (
    "Some files could not be removed. Close the old AddaxAI if it is "
    "running, then try again."
)


class _PurgeState:
    """In-memory state for the legacy-removal background task."""

    def __init__(self) -> None:
        self.in_progress: bool = False
        self.error: str | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Return True if started, False if already running."""
        with self._lock:
            if self.in_progress:
                return False
            self.in_progress = True
            self.error = None
            return True

    def finish(self, error: str | None = None) -> None:
        with self._lock:
            self.in_progress = False
            self.error = error


_purge_state = _PurgeState()


class LegacyInstallStatus(BaseModel):
    """Presence and removal progress in one payload, polled at ~1.5s."""

    found: bool
    version: str | None
    removable_paths: list[str]
    manual_paths: list[str]
    removal_in_progress: bool
    removal_error: str | None


@router.get("/legacy-install", response_model=LegacyInstallStatus)
def get_legacy_install() -> LegacyInstallStatus:
    found = legacy_install.scan()
    return LegacyInstallStatus(
        found=found.found,
        version=found.version,
        removable_paths=[str(p) for p in found.removable],
        manual_paths=[str(p) for p in found.manual],
        removal_in_progress=_purge_state.in_progress,
        removal_error=_purge_state.error,
    )


@router.post("/legacy-install/remove", status_code=status.HTTP_202_ACCEPTED)
async def remove_legacy_install() -> dict[str, str]:
    """
    Delete the legacy install in the background.

    Deleting a legacy tree means hundreds of thousands of small files and
    takes minutes on Windows, so this can't block a request. The frontend
    polls GET /legacy-install until removal_in_progress clears.
    """
    if not _purge_state.start():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A removal is already running.",
        )

    # Same shape as install_env above: blocking work in a thread so the
    # event loop keeps serving the status poll that drives the dialog.
    asyncio.create_task(asyncio.to_thread(_remove_legacy_blocking))
    return {"status": "started"}


def _remove_legacy_blocking() -> None:
    try:
        survivors = legacy_install.remove()
        _purge_state.finish(_RETRY_MESSAGE if survivors else None)
    except Exception as e:
        logger.error(f"Legacy removal failed: {e}", exc_info=True)
        _purge_state.finish(f"Removal failed: {e}")
