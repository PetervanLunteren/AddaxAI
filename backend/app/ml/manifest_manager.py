"""
Model manifest manager.

Following DEVELOPERS.md principles:
- Crash early if configuration is invalid
- Explicit error handling
- Type hints everywhere
"""

import urllib.request
from pathlib import Path
from typing import Any

import yaml

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

    def __init__(self, bundled_path: str | None = None, remote_url: str | None = None):
        """
        Initialize manifest manager.

        Args:
            bundled_path: Path to bundled manifests YAML (relative to app root)
            remote_url: URL to fetch remote manifest updates from
        """
        self.bundled_path = bundled_path or "app/ml/manifests/models.yaml"
        self.remote_url = remote_url or "https://raw.githubusercontent.com/yourusername/addaxai-manifests/main/models.yaml"
        self._cache: dict[str, ModelManifest] | None = None

    def load_manifests(self, force_refresh: bool = False) -> dict[str, ModelManifest]:
        """
        Load all model manifests (bundled + remote).

        Bundled manifests are always available (shipped with app).
        Remote manifests are fetched if available and override bundled ones.

        Args:
            force_refresh: If True, ignore cache and reload from disk/network

        Returns:
            Dictionary mapping model_id to ModelManifest

        Raises:
            FileNotFoundError: If bundled manifest file doesn't exist
            ValueError: If manifest validation fails
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        # Load bundled manifests (always available, crash if missing)
        bundled_data = self._load_bundled()
        logger.info(f"Loaded {len(bundled_data)} models from bundled manifests")

        # Try to fetch remote updates (non-blocking, fail gracefully)
        try:
            remote_data = self._fetch_remote()
            if remote_data:
                logger.info(f"Loaded {len(remote_data)} models from remote manifests")
                # Merge: remote overrides bundled
                merged_data = {**bundled_data, **remote_data}
            else:
                merged_data = bundled_data
        except Exception as e:
            logger.warning(f"Failed to fetch remote manifests (using bundled only): {e}")
            merged_data = bundled_data

        # Validate and convert to ModelManifest objects
        validated_manifests: dict[str, ModelManifest] = {}
        for model_id, model_data in merged_data.items():
            try:
                manifest = ModelManifest(**model_data)
                validated_manifests[model_id] = manifest
            except Exception as e:
                logger.error(f"Invalid manifest for model {model_id}: {e}")
                # Crash early in development, skip in production
                raise ValueError(f"Invalid manifest for model {model_id}: {e}") from e

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

    def _load_bundled(self) -> dict[str, Any]:
        """
        Load bundled manifest YAML file.

        Returns:
            Raw manifest data (not yet validated)

        Raises:
            FileNotFoundError: If bundled manifest doesn't exist
        """
        # Convert relative path to absolute
        manifest_path = Path(__file__).parent / "manifests" / "models.yaml"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Bundled manifest not found: {manifest_path}. "
                f"Application is misconfigured."
            )

        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        # Combine detection and classification models into single dict
        all_models = {}
        if "detection_models" in data:
            all_models.update(data["detection_models"])
        if "classification_models" in data:
            all_models.update(data["classification_models"])

        return all_models

    def _fetch_remote(self, timeout: int = 5) -> dict[str, Any] | None:
        """
        Fetch remote manifest updates.

        Non-blocking - returns None if fetch fails or times out.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Raw manifest data or None if fetch fails
        """
        try:
            logger.info(f"Fetching remote manifests from {self.remote_url}")

            # Fetch remote YAML
            req = urllib.request.Request(
                self.remote_url,
                headers={"User-Agent": "AddaxAI/0.1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw_yaml = response.read().decode("utf-8")

            data = yaml.safe_load(raw_yaml)

            # Combine detection and classification models
            all_models = {}
            if "detection_models" in data:
                all_models.update(data["detection_models"])
            if "classification_models" in data:
                all_models.update(data["classification_models"])

            return all_models

        except Exception as e:
            logger.debug(f"Remote manifest fetch failed: {e}")
            return None
