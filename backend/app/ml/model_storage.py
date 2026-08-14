"""
Model storage manager for downloading and caching model weights from HuggingFace.

Based on proven patterns from streamlit-AddaxAI.
All models (detection and classification) download from HuggingFace repos.

Following DEVELOPERS.md principles:
- Crash early if downloads fail
- Explicit error messages
- Type hints everywhere
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from app.core.config import get_settings
from app.core.job_cancellation import JobCancelledError
from app.core.logging_config import get_logger
from app.ml.hf_downloader import HuggingFaceRepoDownloader
from app.ml.schemas.model_manifest import ModelManifest, resolve_hf_repo
from app.utils.fs_remove import safe_rmtree

logger = get_logger(__name__)

# Seconds to wait on the HuggingFace listing during a staleness check. The
# check runs once per installed model at startup, so an unreachable host
# must fail rather than hang.
_HF_TIMEOUT = 10.0

# Repo files that are never part of a working model install, matched by
# basename anywhere in the repo. Documentation the app never reads,
# .gitattributes which only drives LFS server-side, Finder litter, and
# manifest.json, which is written from models.json by the catalog updater:
# if a repo ever shipped one, the two writers would overwrite each other on
# alternating operations and the model would be permanently "out of date".
_IGNORED_REPO_FILES = frozenset(
    {
        "README.md",
        "LICENSE",
        "LICENSE.md",
        ".gitattributes",
        ".DS_Store",
        "manifest.json",
    }
)


def _clear_downloaded_files(model_dir: Path) -> None:
    """
    Remove everything a download put in `model_dir`, keeping manifest.json.

    manifest.json is the one file in a model directory that no download
    owns: it is written from models.json by the catalog updater and is in
    `_IGNORED_REPO_FILES`, so a download can delete it but never put it
    back. Losing it drops the model out of the catalog until the next
    launch's sync, which the user meets as "Classification model '<id>'
    not found" the next time they create a project, while the weights sit
    on disk. Per-entry `safe_rmtree` so one locked file cannot abort the
    rest of the cleanup.
    """
    if not model_dir.exists():
        return
    for entry in model_dir.iterdir():
        if entry.name == "manifest.json":
            continue
        safe_rmtree(entry)


def git_blob_sha1(path: Path) -> str:
    """
    Git blob SHA-1 of a file: sha1(b"blob <bytelen>\\0" + contents).

    This is exactly what HuggingFace reports as a file's `blob_id`, so a
    local file can be compared to upstream without downloading anything.
    The algorithm is dictated by git's object format, not chosen by us:
    see `find_stale_files`. Not to be confused with `hash_yaml_file` in
    environment_manager.py, which is a plain sha256 over raw bytes and is
    only ever compared against a value this app wrote itself.
    """
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def find_stale_files(model_dir: Path, hf_repo: str) -> list[str] | None:
    """
    Repo-relative paths whose local copy differs from HuggingFace.

    Every non-LFS file in the repo is compared by git blob SHA-1, which
    HuggingFace reports in the same listing, so nothing is downloaded to
    reach a verdict. LFS files are skipped: those are the model weights,
    they are versioned by model_id rather than replaced in place, and
    HuggingFace's blob_id for an LFS file is the hash of the pointer stub
    rather than of the content, so comparing it would report every install
    as stale forever. A file the repo has and the install does not counts
    as stale. Local files the repo does not have are left alone: this
    never proposes a deletion.

    Returns:
        Sorted repo-relative paths, [] when the install matches upstream,
        or None when the question cannot be answered: offline, private or
        missing repo, or an unreadable local file. Never raises, so a
        staleness check cannot take down startup.
    """
    try:
        info = HfApi(endpoint=get_settings().hf_base_url).model_info(
            hf_repo, files_metadata=True, timeout=_HF_TIMEOUT
        )
    except RepositoryNotFoundError:
        # Private or renamed repo. Permanent on this machine, so logging it
        # at warning would repeat the same line on every single launch.
        logger.debug(f"Repo {hf_repo} not accessible, skipping staleness check")
        return None
    except Exception as e:
        logger.warning(f"Could not list {hf_repo} to check for updates: {e}")
        return None

    stale: list[str] = []
    for sibling in info.siblings or []:
        if sibling.lfs:
            continue
        if Path(sibling.rfilename).name in _IGNORED_REPO_FILES:
            continue
        if not sibling.blob_id:
            # Nothing to compare against. Treating it as stale would
            # re-download the file and flag it again on the next launch,
            # leaving an update prompt the user can never clear.
            logger.debug(f"{hf_repo}/{sibling.rfilename} has no blob_id, skipping")
            continue

        local = model_dir / sibling.rfilename
        if not local.is_file():
            stale.append(sibling.rfilename)
            continue
        try:
            digest = git_blob_sha1(local)
        except OSError as e:
            # An unreadable local file means we cannot answer the question
            # at all. Calling it stale would just queue a download that
            # hits the same error.
            logger.warning(f"Could not read {local} to check for updates: {e}")
            return None
        if digest != sibling.blob_id:
            stale.append(sibling.rfilename)

    return sorted(stale)


class ModelStorage:
    """
    Manages model weight downloads and caching from HuggingFace.

    All models download from HF repos to ~/AddaxAI/models/{model_id}/
    """

    def __init__(self, models_dir: Path | None = None):
        """
        Initialize model storage manager.

        Args:
            models_dir: Directory to store model weights (default: settings.models_dir)
        """
        self.models_dir = models_dir or get_settings().models_dir
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
        should_cancel: Callable[[], bool] | None = None,
    ) -> Path:
        """
        Download model weights from HuggingFace if not cached.

        Refreshing an install that is already present is `update_stale_files`,
        which fetches only what actually changed.

        Args:
            manifest: Model manifest
            progress_callback: Optional callback(message, progress) for updates
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

        # Skip if already downloaded.
        if self.check_weights_ready(manifest):
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

            if progress_callback:
                progress_callback("Download complete", 1.0)

            return model_path

        except JobCancelledError:
            # The user asked to stop, so throw the download away and let a
            # later attempt start clean. manifest.json survives: see
            # _clear_downloaded_files.
            logger.info(f"Cleaning up cancelled download at {model_path}")
            _clear_downloaded_files(model_path)
            raise
        except Exception as e:
            # No cleanup on failure, on purpose. Every file is streamed to a
            # `.tmp` sibling and only renamed into place once it is complete
            # and its size matches, so a file at its final path is whole and
            # `download_file` skips it next time. Keeping them makes a retry
            # fetch only what is actually missing.
            #
            # This used to rmtree the directory. That was written for the
            # December 2025 downloader, which streamed straight to the final
            # path and so could leave truncated files behind; the January
            # `.tmp` + atomic rename removed that failure mode and left the
            # wipe doing nothing but harm. On 2026-08-12 one unresolvable
            # 12 KB inference.py deleted a 1.13 GB weights file that had
            # downloaded perfectly, so the retry re-fetched the whole model,
            # and it deleted manifest.json, so the model then did not exist
            # as far as the rest of the app was concerned.
            raise RuntimeError(
                f"Failed to download {manifest.model_id} "
                f"from {hf_repo}: {e}"
            ) from e

    def update_stale_files(self, manifest: ModelManifest) -> list[str]:
        """
        Re-download only the repo files whose local copy differs from
        HuggingFace. Nothing is wiped, nothing is deleted, and the weights
        are never touched, so refreshing a fixed inference.py costs a few
        kilobytes instead of the whole model.

        Returns:
            The sorted repo-relative paths that were refreshed. Empty means
            the install already matched upstream.

        Raises:
            FileNotFoundError: the model is not installed on this machine.
            ConnectionError: upstream could not be reached to decide.
            RuntimeError: a file was found to be stale but failed to download.
        """
        # Raises FileNotFoundError with a message aimed at the user when the
        # model directory or the weights file is absent.
        model_dir = self.get_model_file(manifest).parent
        hf_repo = resolve_hf_repo(manifest.model_id, manifest.hf_repo)

        stale = find_stale_files(model_dir, hf_repo)
        if stale is None:
            raise ConnectionError(f"Could not reach {hf_repo} to check for updates")
        if not stale:
            return []

        downloader = HuggingFaceRepoDownloader(max_workers=4)
        success = downloader.download_repo(
            repo_id=hf_repo,
            local_dir=model_dir,
            revision="main",
            include=set(stale),
            # These files were proven different by content, so the
            # downloader's size-equality skip must not apply: an upstream
            # edit that leaves the byte count unchanged would otherwise be
            # skipped and reported as stale again forever.
            overwrite=True,
        )
        if not success:
            raise RuntimeError(f"Failed to update {manifest.model_id} from {hf_repo}")

        logger.info(
            f"Updated {len(stale)} file(s) for {manifest.model_id}: {', '.join(stale)}"
        )
        return stale

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
