"""
Model manifest manager.

Following DEVELOPERS.md principles:
- Crash early if configuration is invalid
- Explicit error handling
- Type hints everywhere
"""

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)


class ManifestManager:
    """
    Manages model manifests (bundled + remote updates).

    Loads model manifests from bundled YAML file and optionally
    fetches updates from remote source. Remote manifests override
    bundled ones to allow model updates without app updates.
    """

    def __init__(self, models_dir: Path | None = None):
        """
        Initialize manifest manager.

        Args:
            models_dir: Root models directory containing det/ and cls/ subdirectories
        """
        if models_dir is None:
            settings = get_settings()
            models_dir = settings.user_data_dir / "models"
        self.models_dir = models_dir
        self._cache: dict[str, ModelManifest] | None = None

    def load_manifests(self, force_refresh: bool = False) -> dict[str, ModelManifest]:
        """
        Load all model manifests by scanning det/ and cls/ subdirectories.

        Each model directory should contain a manifest.json file.

        Args:
            force_refresh: If True, ignore cache and reload from disk

        Returns:
            Dictionary mapping model_id to ModelManifest

        Raises:
            FileNotFoundError: If models directory doesn't exist
            ValueError: If manifest validation fails
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Models directory not found: {self.models_dir}. "
                f"Application is misconfigured."
            )

        # Scan det/ and cls/ subdirectories for model directories
        validated_manifests: dict[str, ModelManifest] = {}

        for model_type in ["det", "cls"]:
            type_dir = self.models_dir / model_type
            if not type_dir.exists():
                logger.warning(f"Model type directory not found: {type_dir}")
                continue

            # Each subdirectory is a model
            for model_dir in type_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                manifest_path = model_dir / "manifest.json"
                if not manifest_path.exists():
                    logger.warning(f"No manifest.json found in {model_dir}")
                    continue

                try:
                    with open(manifest_path) as f:
                        data = json.load(f)

                    manifest = ModelManifest(**data)
                    validated_manifests[manifest.model_id] = manifest
                    logger.debug(f"Loaded manifest for {manifest.model_id} from {model_dir.name}")

                except Exception as e:
                    logger.error(f"Invalid manifest in {manifest_path}: {e}")
                    # Crash early in development
                    raise ValueError(f"Invalid manifest in {manifest_path}: {e}") from e

        if not validated_manifests:
            logger.warning(f"No valid model manifests found in {self.models_dir}")
            self._cache = {}
            return self._cache

        logger.info(f"Loaded {len(validated_manifests)} model manifests from {self.models_dir}")
        self._cache = validated_manifests
        return self._cache

    def get_model(self, model_id: str) -> ModelManifest:
        """
        Get manifest for specific model.

        Args:
            model_id: Model identifier (e.g., "MDV5A", "EUR-DF-v1-3")

        Returns:
            ModelManifest for the requested model

        Raises:
            ValueError: If model_id is not found
        """
        manifests = self.load_manifests()
        if model_id not in manifests:
            available = ", ".join(manifests.keys())
            raise ValueError(
                f"Unknown model: {model_id}. Available models: {available}"
            )
        return manifests[model_id]

    def get_detection_models(self) -> dict[str, ModelManifest]:
        """Get all detection models."""
        manifests = self.load_manifests()
        return {
            model_id: manifest
            for model_id, manifest in manifests.items()
            if manifest.type == "detection"
        }

    def get_classification_models(self) -> dict[str, ModelManifest]:
        """Get all classification models."""
        manifests = self.load_manifests()
        return {
            model_id: manifest
            for model_id, manifest in manifests.items()
            if manifest.type == "classification"
        }

