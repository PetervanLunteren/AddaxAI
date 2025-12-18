"""
HuggingFace Repository Downloader with Multi-threading.

Adapted from streamlit-AddaxAI's proven downloader.
Optimized multi-threaded downloader for HuggingFace model repositories with:
- Adaptive worker scaling based on connection speed
- Progress tracking via callbacks
- Resume capability for interrupted downloads
- Thread-safe progress updates

Following DEVELOPERS.md principles:
- Crash early if downloads fail
- Explicit error messages
- Type hints everywhere
"""

import os
import time
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class HuggingFaceRepoDownloader:
    """Multi-threaded HuggingFace repository downloader with adaptive scaling."""

    def __init__(self, max_workers: int = 4, chunk_size: int = 8192, timeout: int = 30):
        """
        Initialize the Hugging Face repository downloader.

        Args:
            max_workers: Maximum number of concurrent downloads
            chunk_size: Size of chunks for file downloads (bytes)
            timeout: Request timeout in seconds
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.api = HfApi()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AddaxAI-HuggingFace-Downloader/1.0"})

        # Adaptive scaling parameters
        self.min_workers = 1
        self.max_workers_limit = 16
        self.speed_samples: list[float] = []
        self.max_speed_samples = 10
        self.last_adjustment_time = 0.0
        self.adjustment_interval = 10  # seconds

        # Progress tracking
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.start_time = 0.0
        self.lock = threading.Lock()

    def get_repo_info(self, repo_id: str, revision: str = "main") -> tuple[int, list[dict]]:
        """
        Get repository information including total size and file list.

        Args:
            repo_id: Repository ID (e.g., "Addax-Data-Science/MDV5A")
            revision: Branch or revision to download

        Returns:
            Tuple of (total_size_bytes, files_info_list)

        Raises:
            ValueError: If repository not found
            RuntimeError: If error fetching repository info
        """
        try:
            logger.info(f"Fetching repository info for {repo_id}...")

            # Get repository files
            files = self.api.list_repo_files(
                repo_id=repo_id, revision=revision, repo_type="model"
            )

            # Get detailed file information
            files_info = []
            total_size = 0

            logger.info(f"Analyzing {len(files)} files...")

            for file_path in files:
                try:
                    # Get file info using the API
                    file_info_list = self.api.get_paths_info(
                        repo_id=repo_id,
                        paths=[file_path],
                        revision=revision,
                        repo_type="model",
                    )

                    if file_info_list and hasattr(file_info_list[0], "size") and file_info_list[0].size:
                        file_size = file_info_list[0].size
                        total_size += file_size
                        files_info.append(
                            {
                                "path": file_path,
                                "size": file_size,
                                "url": f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}",
                            }
                        )
                    else:
                        # Add file without size info
                        files_info.append(
                            {
                                "path": file_path,
                                "size": 0,
                                "url": f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}",
                            }
                        )

                except Exception as e:
                    logger.warning(f"Could not get size for {file_path}: {e}")
                    # Add file without size info
                    files_info.append(
                        {
                            "path": file_path,
                            "size": 0,
                            "url": f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}",
                        }
                    )

            return total_size, files_info

        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            raise ValueError(f"Repository not found: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error fetching repository info: {e}") from e

    def update_progress(self, bytes_downloaded: int):
        """Update the downloaded bytes counter thread-safely."""
        with self.lock:
            self.downloaded_bytes += bytes_downloaded

    def measure_download_speed(self, start_time: float, bytes_downloaded: int):
        """Measure and record download speed for adaptive scaling."""
        if bytes_downloaded > 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = bytes_downloaded / elapsed  # bytes per second
                with self.lock:
                    self.speed_samples.append(speed)
                    if len(self.speed_samples) > self.max_speed_samples:
                        self.speed_samples.pop(0)

    def adjust_workers(self):
        """Dynamically adjust the number of workers based on performance."""
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return

        with self.lock:
            if len(self.speed_samples) < 3:
                return

            avg_speed = sum(self.speed_samples) / len(self.speed_samples)
            recent_speed = sum(self.speed_samples[-3:]) / 3

            # If recent speed is significantly lower, reduce workers
            if recent_speed < avg_speed * 0.7 and self.max_workers > self.min_workers:
                self.max_workers = max(self.min_workers, self.max_workers - 1)
                logger.info(f"Reduced workers to {self.max_workers} (slow connection)")

            # If recent speed is good and stable, consider increasing workers
            elif recent_speed > avg_speed * 1.2 and self.max_workers < self.max_workers_limit:
                self.max_workers = min(self.max_workers_limit, self.max_workers + 1)
                logger.info(f"Increased workers to {self.max_workers} (fast connection)")

            self.last_adjustment_time = current_time

    def download_file(self, file_info: dict, local_dir: Path) -> bool:
        """
        Download a single file with progress tracking.

        Args:
            file_info: File information including path, size, and URL
            local_dir: Local directory to save the file

        Returns:
            True if successful, False otherwise
        """
        file_path = file_info["path"]
        file_size = file_info["size"]
        file_url = file_info["url"]

        local_file_path = local_dir / file_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if file already exists and has correct size
        if local_file_path.exists():
            existing_size = local_file_path.stat().st_size
            if existing_size == file_size:
                self.update_progress(file_size)
                return True

        start_time = time.time()
        downloaded = 0

        try:
            with self.session.get(file_url, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()

                with open(local_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            chunk_size = len(chunk)
                            downloaded += chunk_size
                            self.update_progress(chunk_size)

            self.measure_download_speed(start_time, downloaded)
            return True

        except Exception as e:
            logger.error(f"Failed to download {file_path}: {e}")
            # Clean up partial file
            if local_file_path.exists():
                local_file_path.unlink()
            return False

    def download_repo(
        self,
        repo_id: str,
        local_dir: Path,
        progress_callback: Callable[[str, float], None] | None = None,
        revision: str = "main",
    ) -> bool:
        """
        Download entire Hugging Face repository.

        Args:
            repo_id: Repository ID (e.g., "Addax-Data-Science/MDV5A")
            local_dir: Local directory to save files
            progress_callback: Optional callback(message, progress) for updates
            revision: Branch or revision to download

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting download of {repo_id} (revision: {revision})")

            if progress_callback:
                progress_callback(f"Fetching repository info for {repo_id}...", 0.0)

            # Get repository info and total size
            total_size, files_info = self.get_repo_info(repo_id, revision)
            self.total_bytes = total_size
            self.downloaded_bytes = 0
            self.start_time = time.time()

            size_gb = total_size / (1024 * 1024 * 1024)
            logger.info(f"Repository size: {size_gb:.2f} GB ({len(files_info)} files)")

            if progress_callback:
                progress_callback(f"Downloading {len(files_info)} files ({size_gb:.2f} GB)...", 0.05)

            # Create local directory
            local_dir.mkdir(parents=True, exist_ok=True)

            # Download files with thread pool
            successful_downloads = 0
            failed_downloads = 0
            last_progress_update = time.time()
            progress_update_interval = 0.5  # Update progress every 500ms

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all download tasks
                future_to_file = {
                    executor.submit(self.download_file, file_info, local_dir): file_info
                    for file_info in files_info
                }

                # Process completed downloads
                completed = 0

                while future_to_file:
                    # Check for completed downloads
                    completed_futures = [f for f in future_to_file.keys() if f.done()]

                    # Process completed futures
                    for future in completed_futures:
                        file_info = future_to_file.pop(future)
                        try:
                            success = future.result()
                            if success:
                                successful_downloads += 1
                            else:
                                failed_downloads += 1
                        except Exception as e:
                            logger.error(f"Unexpected error downloading {file_info['path']}: {e}")
                            failed_downloads += 1

                        completed += 1

                    # Send periodic progress updates (even while downloading)
                    current_time = time.time()
                    if progress_callback and total_size > 0 and (
                        current_time - last_progress_update >= progress_update_interval or
                        len(completed_futures) > 0  # Also update when files complete
                    ):
                        last_progress_update = current_time
                        overall_progress = 0.05 + (self.downloaded_bytes / total_size) * 0.9

                        # Calculate download speed
                        elapsed = current_time - self.start_time
                        if elapsed > 0:
                            speed_mbps = (self.downloaded_bytes / elapsed) / (1024 * 1024)
                            downloaded_mb = self.downloaded_bytes / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            percentage = (self.downloaded_bytes / total_size) * 100

                            progress_callback(
                                f"{percentage:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB) @ {speed_mbps:.2f} MB/s",
                                overall_progress,
                            )

                    # Periodically adjust workers based on performance
                    self.adjust_workers()

                    # Short sleep to prevent busy waiting
                    time.sleep(0.1)

            # Summary
            logger.info(f"Download completed! Success: {successful_downloads}, Failed: {failed_downloads}")

            if progress_callback:
                progress_callback("Download complete", 1.0)

            return failed_downloads == 0

        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Download failed: {e}", 0.0)
            return False
