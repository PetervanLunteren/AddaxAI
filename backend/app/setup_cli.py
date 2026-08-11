"""
Scripted setup for IT deployments.

`backend --setup [MODEL_ID ...]` runs the same setup the first-launch
wizard runs (base env plus default models), then installs any extra
models by ID, exactly like the in-app install button. IT teams call it
from a deployment or logon script after a silent install so the user's
first click needs no downloads. `backend --list-models` prints the
installable model IDs.

Everything here writes to the invoking user's data directory, so on
shared machines it runs once per user account, not once per machine.

Progress goes to stdout, errors to stderr and the backend log. Exit
code 0 on success (including nothing to do), 1 on failure.
"""

import asyncio
import sys
from collections.abc import Callable

from app.core.config import get_settings
from app.core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _print_progress(message: str, progress: float) -> None:
    print(f"[{progress:4.0%}] {message}", flush=True)


def _model_progress(model_id: str) -> Callable[[str, float], None]:
    return lambda msg, prog: _print_progress(f"{model_id}: {msg}", prog)


def _sync_catalog() -> None:
    """
    Fetch models.json and materialise manifest stubs, the same sync the
    app runs at startup. Best-effort: offline, any manifests already on
    disk still resolve, and a model that stays unknown fails loudly in
    _install_models.
    """
    from app.ml.catalog_updater import ModelCatalogUpdater

    updater = ModelCatalogUpdater(catalog_url=get_settings().model_catalog_url)
    try:
        asyncio.run(updater.sync())
    except Exception as e:
        logger.warning(f"Model catalog sync failed: {e}")
        print(f"Warning: could not refresh the model catalog: {e}", file=sys.stderr)


def _resolve_models(model_ids: list[str]) -> list:
    """
    Turn model IDs into manifests. Raises ValueError naming the unknown
    ID and the valid ones. Called before the base setup so a typo fails
    in seconds, not after a 30-minute env build.
    """
    from app.ml.manifest_manager import ManifestManager

    models_dir = get_settings().models_dir
    manifests = ManifestManager(models_dir)
    return [manifests.get_model(model_id) for model_id in model_ids]


def _install_models(resolved: list) -> None:
    from app.ml.environment_manager import EnvironmentManager
    from app.ml.model_storage import ModelStorage

    models_dir = get_settings().models_dir
    storage = ModelStorage(models_dir)
    em = EnvironmentManager()

    for manifest in resolved:
        if storage.check_weights_ready(manifest):
            _print_progress(f"{manifest.model_id}: weights already present", 1.0)
        else:
            storage.download_weights(manifest, _model_progress(manifest.model_id))
        # No-ops when the env is already valid on disk.
        em.get_or_create_env(manifest, _model_progress(manifest.model_id))


def _list_models() -> int:
    from app.ml.manifest_manager import ManifestManager

    _sync_catalog()
    models_dir = get_settings().models_dir
    manifests = ManifestManager(models_dir).load_manifests()
    if not manifests:
        print(
            "No models found. Check the internet connection and try again.",
            file=sys.stderr,
        )
        return 1
    for m in sorted(
        manifests.values(), key=lambda m: (m.model_category or "", m.model_id)
    ):
        print(f"{m.model_id:24} {m.model_category or '?':16} {m.friendly_name}")
    return 0


def run_cli(argv: list[str]) -> int:
    """
    Entry point, dispatched from run_server.py. `argv` is sys.argv[1:];
    every argument that is not a flag is treated as a model ID.
    """
    setup_logging()
    # Keep the console clean for scripts: progress lines only. The full
    # log, including tracebacks, still lands in backend.log. The
    # dev-mode console handler would otherwise interleave every log
    # line with the output and print each failure twice.
    import logging

    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            root.removeHandler(handler)

    if "--list-models" in argv:
        return _list_models()

    model_ids = [a for a in argv if not a.startswith("-")]

    try:
        from fastapi import HTTPException

        from app.api.routers.setup import (
            _check_disk_space,
            run_setup,
            setup_complete,
        )

        # Resolve the requested models before the (potentially long)
        # base setup so a typo in the script fails in seconds.
        resolved = []
        if model_ids:
            _sync_catalog()
            resolved = _resolve_models(model_ids)

        if setup_complete():
            print("Base setup already present", flush=True)
        else:
            # Same pre-flight the wizard runs: fail fast with a clear
            # message instead of a cryptic OSError mid-download.
            try:
                _check_disk_space(get_settings().user_data_dir)
            except HTTPException as e:
                print(f"Error: {e.detail}", file=sys.stderr, flush=True)
                return 1
            print("Running base setup (env + default models)...", flush=True)
            run_setup(_print_progress)

        if resolved:
            _install_models(resolved)
    except Exception as e:
        logger.error(f"Scripted setup failed: {e}", exc_info=True)
        print(f"Setup failed: {e}", file=sys.stderr, flush=True)
        return 1

    print("Setup complete", flush=True)
    return 0
