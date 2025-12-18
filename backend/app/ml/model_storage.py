"""
Model storage manager for downloading and caching model weights from HuggingFace.

Based on proven patterns from streamlit-AddaxAI.
All models (detection and classification) download from HuggingFace repos.

Following DEVELOPERS.md principles:
- Crash early if downloads fail
- Explicit error messages
- Type hints everywhere
"""

from pathlib import Path
from typing import Callable

from app.core.logging_config import get_logger
from app.ml.hf_downloader import HuggingFaceRepoDownloader
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)


class ModelStorage:
    """
    Manages model weight downloads and caching from HuggingFace.

    All models download from HF repos to ~/AddaxAI/models/{model_id}/
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
        # Model is in models/det/{model_id}/ or models/cls/{model_id}/
        model_type = "det" if manifest.type == "detection" else "cls"
        model_path = self.models_dir / model_type / manifest.model_id
        model_file = model_path / manifest.model_fname

        # Check if model file exists
        return model_file.exists()

    def download_weights(
        self,
        manifest: ModelManifest,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Path:
        """
        Download model weights from HuggingFace if not cached.

        Args:
            manifest: Model manifest
            progress_callback: Optional callback(message, progress) for updates

        Returns:
            Path to model directory

        Raises:
            RuntimeError: If download fails
        """
        # Model is in models/det/{model_id}/ or models/cls/{model_id}/
        model_type = "det" if manifest.type == "detection" else "cls"
        model_path = self.models_dir / model_type / manifest.model_id

        # Skip if already exists
        if self.check_weights_ready(manifest):
            logger.info(f"Model {manifest.model_id} already cached at {model_path}")
            if progress_callback:
                progress_callback("Model already cached", 1.0)
            return model_path

        # Determine HF repo
        hf_repo = manifest.hf_repo or f"Addax-Data-Science/{manifest.model_id}"
        logger.info(f"Downloading {hf_repo} to {model_path}")

        if progress_callback:
            progress_callback(f"Downloading {manifest.friendly_name} from HuggingFace...", 0.0)

        try:
            # Download using multi-threaded downloader
            downloader = HuggingFaceRepoDownloader(max_workers=4)
            success = downloader.download_repo(
                repo_id=hf_repo,
                local_dir=model_path,
                progress_callback=progress_callback,
                revision="main",
            )

            if not success:
                raise RuntimeError(f"Download failed for {hf_repo}")

            # Verify the model file exists
            model_file = model_path / manifest.model_fname
            if not model_file.exists():
                raise RuntimeError(
                    f"Model file not found after download: {manifest.model_fname}\n"
                    f"Expected at: {model_file}\n"
                    f"Downloaded files: {list(model_path.rglob('*'))}"
                )

            logger.info(f"Downloaded {manifest.model_id} to {model_path}")

            if progress_callback:
                progress_callback("Download complete", 1.0)

            return model_path

        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                import shutil

                logger.warning(f"Cleaning up partial download at {model_path}")
                shutil.rmtree(model_path)

            raise RuntimeError(f"Failed to download {manifest.model_id} from {hf_repo}: {e}") from e

    def get_model_path(self, manifest: ModelManifest) -> Path:
        """
        Get path to model directory.

        Args:
            manifest: Model manifest

        Returns:
            Path to model directory (models/det/{model_id}/ or models/cls/{model_id}/)

        Raises:
            FileNotFoundError: If model not downloaded
        """
        model_type = "det" if manifest.type == "detection" else "cls"
        model_path = self.models_dir / model_type / manifest.model_id
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model {manifest.model_id} not found at {model_path}. " f"Please download it first."
            )

        return model_path

    def get_model_file(self, manifest: ModelManifest) -> Path:
        """
        Get path to model weight file.

        Args:
            manifest: Model manifest

        Returns:
            Path to model file (e.g., .pt, .pth)

        Raises:
            FileNotFoundError: If model file not found
        """
        model_path = self.get_model_path(manifest)
        model_file = model_path / manifest.model_fname

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {manifest.model_fname}\n" f"Expected at: {model_file}"
            )

        return model_file

    def get_weights_size(self, manifest: ModelManifest) -> float | None:
        """
        Get size of downloaded weights in MB.

        Args:
            manifest: Model manifest

        Returns:
            Size in MB or None if not downloaded
        """
        model_path = self.models_dir / manifest.model_id
        if not model_path.exists():
            return None

        # Calculate directory size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

        return total_size / (1024 * 1024)  # Convert to MB
