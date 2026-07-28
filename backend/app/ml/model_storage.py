"""
Model storage manager for downloading and caching model weights from HuggingFace.

Based on proven patterns from streamlit-AddaxAI.
All models (detection and classification) download from HuggingFace repos.

Following DEVELOPERS.md principles:
- Crash early if downloads fail
- Explicit error messages
- Type hints everywhere
"""

import json
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi

from app.core.job_cancellation import JobCancelledError
from app.core.logging_config import get_logger
from app.ml.hf_downloader import HuggingFaceRepoDownloader
from app.ml.schemas.model_manifest import ModelManifest, resolve_hf_repo

logger = get_logger(__name__)


def _record_hf_revision_sha(manifest_path: Path, sha: str) -> None:
    """
    Persist the HF commit SHA into the local manifest.json so a later
    drift check can compare on-disk vs upstream. Read-modify-write on a
    file the catalog updater also touches; both call sites update the
    file rarely so the lack of locking is fine for now.

    Never raises; failure here just means drift detection won't fire
    for this model until the next successful re-download. Logged at
    warning so the diagnostic ZIP records the miss.
    """
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        if data.get("hf_revision_sha") == sha:
            return
        data["hf_revision_sha"] = sha
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Recorded hf_revision_sha={sha[:12]}... in {manifest_path}"
        )
    except Exception as e:
        logger.warning(
            f"Failed to record hf_revision_sha into {manifest_path}: {e}"
        )


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
        Check if model files are downloaded and ready for inference.

        For embedding models that load their architecture via
        torch.hub.load(..., source="local"), the HF repo also ships the
        dinov2/ source and a hubconf.py next to the .pth. A pre-upgrade
        install would have the .pth but be missing those, which manifests
        at inference time as a FileNotFoundError. Treating that state as
        "not ready" routes the user through the normal Prepare-model UI
        instead, where the HF downloader fills in the missing files
        (the existing .pth is recognized by size and skipped).

        Args:
            manifest: Model manifest

        Returns:
            True if all required files are present, False if download needed
        """
        # Model is in models/det/{model_id}/ or models/cls/{model_id}/
        # Use model_category (set by ManifestManager based on directory) to determine path
        model_type = {"detection": "det", "classification": "cls", "embedding": "emb"}[
            manifest.model_category
        ]
        model_path = self.models_dir / model_type / manifest.model_id
        model_file = model_path / manifest.model_fname

        if not model_file.exists():
            return False

        # Architecture source check: only applies to models that load via
        # torch.hub.load(source="local"). Other models load their
        # architecture from PyPI packages in the analysis env and don't
        # ship source alongside the weights.
        if manifest.torch_hub_model and not (model_path / "hubconf.py").is_file():
            return False

        return True

    def download_weights(
        self,
        manifest: ModelManifest,
        progress_callback: Callable[[str, float], None] | None = None,
        force: bool = False,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Path:
        """
        Download model weights from HuggingFace if not cached.

        Args:
            manifest: Model manifest
            progress_callback: Optional callback(message, progress) for updates
            force: If True, wipe the model directory (preserving manifest.json)
                before downloading. Used by the drift-redownload flow when the
                upstream HF revision moved past the locally recorded SHA.
            should_cancel: Optional predicate polled while downloading; when it
                returns True the partial download is removed and
                JobCancelledError propagates to the caller.

        Returns:
            Path to model directory

        Raises:
            RuntimeError: If download fails
            JobCancelledError: If cancelled via should_cancel
        """
        # Model is in models/det/{model_id}/ or models/cls/{model_id}/
        # Use model_category (set by ManifestManager based on directory) to determine path
        model_type = {"detection": "det", "classification": "cls", "embedding": "emb"}[
            manifest.model_category
        ]
        model_path = self.models_dir / model_type / manifest.model_id

        if force and model_path.exists():
            # Wipe everything except manifest.json and the weights file
            # itself. manifest.json must survive so the catalog stub is
            # not lost. The weights file must survive so the setup-status
            # check (`_models_present` in routers/setup.py) keeps
            # returning True while the redownload runs; otherwise the
            # SetupGate sees `ready=false` mid-download and redirects the
            # user to the first-run wizard. The HF downloader is
            # size-checked per file (hf_downloader.py:212-217), so a kept
            # weights file is re-fetched only if its on-disk size differs
            # from the upstream size.
            keep = {"manifest.json", manifest.model_fname}
            logger.info(
                f"Force re-download: wiping cached files at {model_path} "
                f"(keeping {sorted(keep)})"
            )
            for child in model_path.iterdir():
                if child.name in keep:
                    continue
                try:
                    if child.is_dir():
                        import shutil
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except OSError as e:
                    logger.warning(
                        f"Could not remove {child} during force re-download: {e}"
                    )

        # Skip if already exists (and not forcing).
        if not force and self.check_weights_ready(manifest):
            logger.info(f"Model {manifest.model_id} already cached at {model_path}")
            if progress_callback:
                progress_callback("Model already cached", 1.0)
            return model_path

        # Determine HF repo
        hf_repo = resolve_hf_repo(manifest.model_id, manifest.hf_repo)
        logger.info(f"Downloading {hf_repo} to {model_path}")

        if progress_callback:
            progress_callback(
                f"Downloading {manifest.friendly_name} "
                f"from HuggingFace...",
                0.0,
            )

        try:
            # Download using multi-threaded downloader
            downloader = HuggingFaceRepoDownloader(max_workers=4)
            success = downloader.download_repo(
                repo_id=hf_repo,
                local_dir=model_path,
                progress_callback=progress_callback,
                revision="main",
                should_cancel=should_cancel,
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

            # Record the HF commit SHA we just downloaded so a later
            # ModelCatalogUpdater.sync() can spot drift when the
            # upstream repo moves. Best-effort: any HF API failure here
            # is logged and ignored. Without this, drift detection
            # silently no-ops for this model.
            try:
                info = HfApi().model_info(hf_repo)
                sha = getattr(info, "sha", None)
                if sha:
                    _record_hf_revision_sha(model_path / "manifest.json", sha)
                else:
                    logger.warning(
                        f"HfApi.model_info({hf_repo}) returned no sha attribute"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to fetch HF revision SHA for {hf_repo} "
                    f"after download: {e}"
                )

            if progress_callback:
                progress_callback("Download complete", 1.0)

            return model_path

        except JobCancelledError:
            # Cancelled mid-download: drop the partial directory so a later
            # retry starts clean, then propagate so the worker reports
            # cancellation rather than a failure.
            if model_path.exists():
                import shutil

                logger.info(f"Cleaning up cancelled download at {model_path}")
                shutil.rmtree(model_path)
            raise
        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                import shutil

                logger.warning(f"Cleaning up partial download at {model_path}")
                shutil.rmtree(model_path)

            raise RuntimeError(
                f"Failed to download {manifest.model_id} "
                f"from {hf_repo}: {e}"
            ) from e

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
        # Use model_category (set by ManifestManager based on directory) to determine path
        model_type = {"detection": "det", "classification": "cls", "embedding": "emb"}[
            manifest.model_category
        ]
        model_path = self.models_dir / model_type / manifest.model_id
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model {manifest.model_id} not found at {model_path}. "
                f"Please download it first."
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
        # Use model_category (set by ManifestManager based on directory) to determine path
        model_type = {"detection": "det", "classification": "cls", "embedding": "emb"}[
            manifest.model_category
        ]
        model_path = self.models_dir / model_type / manifest.model_id
        if not model_path.exists():
            return None

        # Calculate directory size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

        return total_size / (1024 * 1024)  # Convert to MB
