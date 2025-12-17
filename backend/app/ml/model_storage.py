"""
Model storage manager for downloading and caching model weights.

Following DEVELOPERS.md principles:
- Crash early if downloads fail
- Explicit error messages
- Type hints everywhere
"""

from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

from app.core.logging_config import get_logger
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)


class ModelStorage:
    """
    Manages model weight downloads and caching.

    MegaDetector: Uses built-in package downloads (no manual storage needed)
    Classification: Downloads full HF repos to ~/AddaxAI/models/{model_id}/
    """

    def __init__(self, models_dir: Path | None = None):
        """
        Initialize model storage manager.

        Args:
            models_dir: Directory to store model weights (default: ~/AddaxAI/models)
        """
        user_data_dir = Path.home() / "AddaxAI"
        self.models_dir = models_dir or (user_data_dir / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def check_weights_ready(self, manifest: ModelManifest) -> bool:
        """
        Check if model weights are downloaded and ready.

        Args:
            manifest: Model manifest

        Returns:
            True if weights are ready, False if download needed
        """
        if manifest.type == "detection":
            # MegaDetector: Check if package is in the manifest
            # Weights will be downloaded by the package itself
            return manifest.package is not None

        elif manifest.type == "classification":
            # Classification: Check if local directory exists
            model_path = self.models_dir / manifest.model_id
            return model_path.exists() and self._validate_classification_model(model_path)

        return False

    def _validate_classification_model(self, model_path: Path) -> bool:
        """
        Validate classification model directory has required files.

        Args:
            model_path: Path to model directory

        Returns:
            True if valid, False otherwise
        """
        # Check for common model files
        required_files = ["config.json"]  # Most HF models have this
        optional_files = ["taxonomy.csv", "model.pt", "model.pth", "pytorch_model.bin"]

        # At least config.json should exist
        has_config = (model_path / "config.json").exists()

        # At least one model weight file should exist
        has_weights = any((model_path / f).exists() for f in optional_files)

        return has_config or has_weights

    def download_weights(
        self,
        manifest: ModelManifest,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Path:
        """
        Download model weights if not cached.

        Args:
            manifest: Model manifest
            progress_callback: Optional callback(message, progress) for updates

        Returns:
            Path to model directory

        Raises:
            RuntimeError: If download fails
        """
        if manifest.type == "detection":
            # MegaDetector: No manual download needed
            if progress_callback:
                progress_callback("MegaDetector will download on first use", 1.0)
            return self.models_dir  # Return a dummy path

        elif manifest.type == "classification":
            return self._download_classification_weights(manifest, progress_callback)

        else:
            raise ValueError(f"Unknown model type: {manifest.type}")

    def _download_classification_weights(
        self,
        manifest: ModelManifest,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Path:
        """
        Download classification model from HuggingFace.

        Args:
            manifest: Model manifest
            progress_callback: Optional progress callback

        Returns:
            Path to downloaded model directory

        Raises:
            RuntimeError: If download fails
        """
        if not manifest.weights_url:
            raise ValueError(f"No weights_url specified for {manifest.model_id}")

        model_path = self.models_dir / manifest.model_id

        # Skip if already exists
        if model_path.exists() and self._validate_classification_model(model_path):
            logger.info(f"Model {manifest.model_id} already cached at {model_path}")
            if progress_callback:
                progress_callback("Model already cached", 1.0)
            return model_path

        # Parse HF repo from weights_url
        # Expected format: https://huggingface.co/{repo_id}
        if "huggingface.co" not in manifest.weights_url:
            raise ValueError(
                f"Invalid HuggingFace URL: {manifest.weights_url}. "
                f"Expected format: https://huggingface.co/{{repo_id}}"
            )

        # Extract repo_id from URL
        # e.g., https://huggingface.co/user/repo -> user/repo
        parts = manifest.weights_url.rstrip("/").split("huggingface.co/")
        if len(parts) < 2:
            raise ValueError(f"Could not parse repo_id from {manifest.weights_url}")

        repo_id = parts[1]
        logger.info(f"Downloading {repo_id} to {model_path}")

        if progress_callback:
            progress_callback(f"Downloading {manifest.friendly_name} from HuggingFace...", 0.1)

        try:
            # Download entire repo using huggingface_hub
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,  # Use actual files, not symlinks
            )

            logger.info(f"Downloaded {manifest.model_id} to {downloaded_path}")

            if progress_callback:
                progress_callback("Download complete", 1.0)

            return Path(downloaded_path)

        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)

            raise RuntimeError(
                f"Failed to download {manifest.model_id} from {repo_id}: {e}"
            ) from e

    def get_model_path(self, manifest: ModelManifest) -> Path:
        """
        Get path to model directory.

        Args:
            manifest: Model manifest

        Returns:
            Path to model directory

        Raises:
            FileNotFoundError: If model not downloaded
        """
        if manifest.type == "detection":
            # MegaDetector: No local path needed
            return self.models_dir

        model_path = self.models_dir / manifest.model_id
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model {manifest.model_id} not found at {model_path}. "
                f"Please download it first."
            )

        return model_path

    def get_weights_size(self, manifest: ModelManifest) -> float | None:
        """
        Get size of downloaded weights in MB.

        Args:
            manifest: Model manifest

        Returns:
            Size in MB or None if not downloaded
        """
        if manifest.type == "detection":
            return None  # MegaDetector size unknown

        model_path = self.models_dir / manifest.model_id
        if not model_path.exists():
            return None

        # Calculate directory size
        total_size = sum(
            f.stat().st_size for f in model_path.rglob("*") if f.is_file()
        )

        return total_size / (1024 * 1024)  # Convert to MB
