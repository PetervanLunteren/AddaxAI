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

# Default models we bundle. Each entry: (category, model_id, weight_fname).
# The weight filename is the marker we check at the destination to decide
# whether the model is "already installed" — see _is_already_installed.
# Checking the weight (not manifest.json) avoids a race with the catalog
# updater, which writes a stub manifest.json into the same directory as
# part of its own startup task.
_DEFAULT_MODELS: tuple[tuple[str, str, str], ...] = (
    ("det", "MD5A-0-0", "md_v5a.0.0.pt"),
    ("emb", "DINOV2-VITB14", "dinov2_vitb14_pretrain.pth"),
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

    for category, model_id, weight_fname in _DEFAULT_MODELS:
        src_dir = bundle / category / model_id
        dst_dir = user_models_dir / category / model_id

        if not src_dir.is_dir():
            logger.warning(
                f"Bundled model missing in app bundle: {category}/{model_id}"
            )
            continue

        # Skip if the user already has the WEIGHT file. The previous
        # version of this check looked for manifest.json, but the catalog
        # updater writes a stub manifest into the same directory as part
        # of its own startup task, racing with this hook and making it
        # think the model was already installed when only the manifest
        # was. Check the actual heavy file instead — if it's present, we
        # really are done; if not, copy.
        if (dst_dir / weight_fname).is_file():
            counts["skipped"] += 1
            logger.debug(f"User already has {category}/{model_id}; skipping")
            continue

        # The directory may already exist (catalog updater stub) so we
        # can't use shutil.copytree on the dst_dir directly. Copy each
        # file individually instead — overwrites the stub manifest.json
        # with the real one from the bundle as a side benefit.
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            for src_file in src_dir.iterdir():
                if src_file.is_file():
                    shutil.copy2(src_file, dst_dir / src_file.name)
            counts["copied"] += 1
            logger.info(f"Installed bundled model: {category}/{model_id}")
        except Exception as e:
            logger.error(
                f"Failed to install bundled model {category}/{model_id}: {e}",
                exc_info=True,
            )

    return counts
