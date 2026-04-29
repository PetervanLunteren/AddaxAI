"""
Copy default model weights from the PyInstaller bundle into the user data
directory on app startup.

The DMG ships MDv5A and DINOv2-B as embedded data so a fresh install can
run a default analysis fully offline (after the one-time setup wizard
installs env-addaxai-base). The weights live read-only inside the bundle;
the rest of the app expects them under ~/AddaxAI/models/, so this service
copies them on first launch.

Idempotent: if a model file is already present at the destination, that
model is skipped. Never overwrites user data.

Silently no-ops in development (no PyInstaller bundle, no source dir).
"""

import shutil
import sys
from pathlib import Path

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Default models we bundle. Each entry maps a directory name inside the
# bundle to a (category, model_id) tuple matching the layout under
# ~/AddaxAI/models/{category}/{model_id}/.
_DEFAULT_MODELS: tuple[tuple[str, str], ...] = (
    ("det", "MD5A-0-0"),
    ("emb", "DINOV2-VITB14"),
)


def _bundle_root() -> Path | None:
    """
    Return the path to the bundled-models directory inside the PyInstaller
    bundle, or None if not running from a bundle.
    """
    if not getattr(sys, "frozen", False):
        return None
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    candidate = Path(meipass) / "bundled_models"
    return candidate if candidate.exists() else None


def install_bundled_models(user_models_dir: Path) -> dict[str, int]:
    """
    Copy any bundled default models into user_models_dir if missing.

    Args:
        user_models_dir: ~/AddaxAI/models or equivalent. Created if absent.

    Returns:
        Dict with counts: {"copied": int, "skipped": int}.
    """
    counts = {"copied": 0, "skipped": 0}

    bundle = _bundle_root()
    if bundle is None:
        logger.debug("No bundled-models directory; nothing to install")
        return counts

    user_models_dir.mkdir(parents=True, exist_ok=True)

    for category, model_id in _DEFAULT_MODELS:
        src_dir = bundle / category / model_id
        dst_dir = user_models_dir / category / model_id

        if not src_dir.is_dir():
            logger.warning(
                f"Bundled model missing in app bundle: {category}/{model_id}"
            )
            continue

        # Skip if the user already has this model installed. We only check
        # the manifest; a partial install (manifest present, weights gone)
        # is treated as installed because users may legitimately delete
        # weights to save space and re-download via the catalog.
        if (dst_dir / "manifest.json").is_file():
            counts["skipped"] += 1
            logger.debug(f"User already has {category}/{model_id}; skipping")
            continue

        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(src_dir, dst_dir)
            counts["copied"] += 1
            logger.info(f"Installed bundled model: {category}/{model_id}")
        except Exception as e:
            logger.error(
                f"Failed to install bundled model {category}/{model_id}: {e}",
                exc_info=True,
            )

    return counts
