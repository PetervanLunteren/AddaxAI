"""
HuggingFace Repository Downloader with Multi-threading.

Adapted from streamlit-AddaxAI's proven downloader.
Optimized multi-threaded downloader for HuggingFace model repositories with:
- Adaptive worker scaling based on connection speed
- Progress tracking via callbacks
- Per-file retries, and a skip for files already complete on disk, so a
  retried repo download only fetches what is still missing. There is no
  resume within a file: an interrupted file restarts from byte 0.
- Thread-safe progress updates

Following DEVELOPERS.md principles:
- Crash early if downloads fail
- Explicit error messages
- Type hints everywhere
"""

import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from app.core.config import get_settings
from app.core.job_cancellation import JobCancelledError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# HuggingFace throttles each connection to a few MB/s, so a single big
# weights file downloads far below a fast link's capacity. Files at least
# this large are split across _PARALLEL_CONNECTIONS connections, which on
# a fast link is several times faster. Smaller files (configs, taxonomy)
# are not worth the extra requests and take the plain single path.
_PARALLEL_MIN_BYTES = 16 * 1024 * 1024
_PARALLEL_CONNECTIONS = 4

# Attempts per file before the repo download is called failed. The session
# carries urllib3's default Retry(total=0), so before this one transient
# failure on any single file killed the whole download: on 2026-08-12 a
# 12 KB inference.py could not resolve huggingface.co while the other four
# files in the same repo downloaded fine in the same second, and the 1.13 GB
# already on disk was thrown away. Files at or above _PARALLEL_MIN_BYTES
# effectively had a second chance already, since a failed range falls back
# to a single connection; this gives every file the same, plus the short
# pause a momentary resolver failure needs.
_FILE_ATTEMPTS = 3


class HuggingFaceRepoDownloader:
    """Multi-threaded HuggingFace repository downloader with adaptive scaling."""

    def __init__(self, max_workers: int = 4, chunk_size: int = 1024 * 1024, timeout: int = 30):
        """
        Initialize the Hugging Face repository downloader.

        Args:
            max_workers: Maximum number of concurrent file downloads. This
                parallelises across FILES; a single big weights file is
                instead split across connections inside download_file.
            chunk_size: Read size per iteration (bytes). 1 MiB, not the old
                8 KiB, and this matters specifically for the parallel path:
                every chunk takes the shared progress lock, so with 8 KiB
                the four range threads spent their time contending on that
                lock and collapsed back to single-connection speed (~2.5 vs
                ~14 MB/s measured). 1 MiB keeps the lock-acquisition count
                low enough that the connections actually run in parallel.
                For a single connection the rate is network-bound either way.
            timeout: Request timeout in seconds
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.timeout = timeout
        # Mirror support (mainland China): both the API metadata calls
        # and the direct download URLs must go through the endpoint, or
        # the mirror only covers half the traffic.
        self.endpoint = (
            get_settings().hf_endpoint or "https://huggingface.co"
        ).rstrip("/")
        self.api = HfApi(endpoint=self.endpoint)
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

    def get_repo_info(
        self,
        repo_id: str,
        revision: str = "main",
        include: set[str] | None = None,
    ) -> tuple[int, list[dict]]:
        """
        Get repository information including total size and file list.

        Args:
            repo_id: Repository ID (e.g., "Addax-Data-Science/MDV5A")
            revision: Branch or revision to download
            include: Only describe these repo-relative paths. None means the
                whole repo.

        Returns:
            Tuple of (total_size_bytes, files_info_list)

        Raises:
            ValueError: If repository not found
            RuntimeError: If error fetching repository info
        """
        try:
            logger.info(f"Fetching repository info for {repo_id}...")

            # Get repository files
            files = self.api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")

            if include is not None:
                # Filter before the loop below, which costs one HTTP call per
                # file. Updating a single file in a repo that vendors a whole
                # source tree must not pay for all ~170 of its files.
                files = [f for f in files if f in include]

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

                    hf_url = (
                        f"{self.endpoint}/{repo_id}"
                        f"/resolve/{revision}/{file_path}"
                    )
                    if (
                        file_info_list
                        and hasattr(file_info_list[0], "size")
                        and file_info_list[0].size
                    ):
                        file_size = file_info_list[0].size
                        total_size += file_size
                        files_info.append(
                            {
                                "path": file_path,
                                "size": file_size,
                                "url": hf_url,
                            }
                        )
                    else:
                        # Add file without size info
                        files_info.append(
                            {
                                "path": file_path,
                                "size": 0,
                                "url": hf_url,
                            }
                        )

                except Exception as e:
                    logger.warning(f"Could not get size for {file_path}: {e}")
                    # Add file without size info
                    hf_url = (
                        f"{self.endpoint}/{repo_id}"
                        f"/resolve/{revision}/{file_path}"
                    )
                    files_info.append(
                        {
                            "path": file_path,
                            "size": 0,
                            "url": hf_url,
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

    def download_file(
        self,
        file_info: dict,
        local_dir: Path,
        should_cancel: Callable[[], bool] | None = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Download a single file with progress tracking.
        Downloads to a .tmp file first, then renames atomically on success.

        Args:
            file_info: File information including path, size, and URL
            local_dir: Local directory to save the file
            should_cancel: Optional predicate polled between chunks; when it
                returns True the download aborts, the partial .tmp file is
                removed, and the method returns False.
            overwrite: Fetch the file even when a local file of the same size
                is already there. Needed by callers that decided by content
                that the local copy is wrong.

        Returns:
            True if successful, False otherwise
        """
        file_path = file_info["path"]
        file_size = file_info["size"]
        file_url = file_info["url"]

        local_file_path = local_dir / file_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if file already exists and has correct size. `overwrite`
        # defeats this because size equality does not mean the contents
        # match: an upstream edit of the same byte length would otherwise be
        # skipped by the very call that came to replace it.
        if (
            not overwrite
            and local_file_path.exists()
            and local_file_path.stat().st_size == file_size
        ):
            self.update_progress(file_size)
            return True

        # Download to temporary file first
        temp_file_path = local_file_path.with_suffix(local_file_path.suffix + ".tmp")

        for attempt in range(1, _FILE_ATTEMPTS + 1):
            start_time = time.time()
            try:
                if file_size >= _PARALLEL_MIN_BYTES and self._supports_range(file_url):
                    downloaded = self._download_ranges(
                        file_url, temp_file_path, file_size, should_cancel
                    )
                else:
                    downloaded = self._download_stream(
                        file_url, temp_file_path, should_cancel
                    )

                if downloaded is None:  # cancelled mid-download
                    return False

                # Verify size matches expected
                if file_size > 0 and temp_file_path.stat().st_size != file_size:
                    actual = temp_file_path.stat().st_size
                    raise ValueError(
                        f"Downloaded file size mismatch: "
                        f"expected {file_size}, got {actual}"
                    )

                # Atomic move to the final location, only after a successful
                # download. Path.replace (not rename) because when the file is
                # being re-downloaded the destination already exists, and on
                # Windows rename() over an existing file raises WinError 183;
                # replace() overwrites atomically on both POSIX and Windows.
                temp_file_path.replace(local_file_path)

                self.measure_download_speed(start_time, downloaded)
                return True

            except Exception as e:
                # Un-count what this attempt wrote before dropping it, or the
                # retry counts those bytes twice and the progress bar runs
                # past 100%. Measured off the partial file so this stays a
                # delta: the counter is shared with every other file's
                # threads, so it can never be restored to a snapshot.
                partial = (
                    temp_file_path.stat().st_size if temp_file_path.exists() else 0
                )
                if partial:
                    with self.lock:
                        self.downloaded_bytes = max(0, self.downloaded_bytes - partial)
                temp_file_path.unlink(missing_ok=True)

                if attempt == _FILE_ATTEMPTS:
                    logger.error(
                        f"Failed to download {file_path} after "
                        f"{attempt} attempt(s): {e}"
                    )
                    return False
                if should_cancel is not None and should_cancel():
                    logger.info(f"Not retrying {file_path}: cancelled")
                    return False
                logger.warning(
                    f"Attempt {attempt}/{_FILE_ATTEMPTS} for {file_path} failed "
                    f"({e}); retrying in {attempt}s"
                )
                time.sleep(attempt)  # 1 s, then 2 s

        return False

    def _download_stream(
        self,
        file_url: str,
        temp_file_path: Path,
        should_cancel: Callable[[], bool] | None,
    ) -> int | None:
        """
        Download a whole file over one connection into temp_file_path.

        Returns the number of bytes written, or None if cancelled (the
        partial file is removed in that case).
        """
        downloaded = 0
        with self.session.get(file_url, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            with open(temp_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    # Bail promptly on cancel so the worker's executor
                    # shutdown doesn't block on a multi-GB transfer.
                    if should_cancel is not None and should_cancel():
                        f.close()
                        temp_file_path.unlink(missing_ok=True)
                        return None
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.update_progress(len(chunk))
        return downloaded

    def _supports_range(self, file_url: str) -> bool:
        """
        True if the server answers a range request with 206 Partial Content.

        A one-byte probe. On any error, assume no and fall back to a single
        connection rather than risk four workers each pulling the whole file.
        """
        try:
            with self.session.get(
                file_url,
                headers={"Range": "bytes=0-0"},
                stream=True,
                timeout=self.timeout,
            ) as r:
                return r.status_code == 206
        except Exception:
            return False

    def _download_ranges(
        self,
        file_url: str,
        temp_file_path: Path,
        file_size: int,
        should_cancel: Callable[[], bool] | None,
    ) -> int | None:
        """
        Download one file over _PARALLEL_CONNECTIONS connections at once.

        The file is split into that many contiguous byte ranges, each pulled
        by its own thread into a `.partN` file, then the parts are joined in
        order. Separate part files (rather than seeking into one shared
        handle) keep this safe on Windows, where multiple write handles to
        the same file are not reliably allowed.

        Returns file_size on success, None if cancelled. If any range fails
        the whole file is retried on a single connection, so a flaky range
        still completes.
        """
        n = _PARALLEL_CONNECTIONS
        step = file_size // n
        parts = [
            temp_file_path.with_name(f"{temp_file_path.name}.part{i}")
            for i in range(n)
        ]
        # Contiguous, non-overlapping ranges; the last one runs to the end so
        # integer division leaves no gap.
        bounds = [
            (i, i * step, file_size - 1 if i == n - 1 else (i + 1) * step - 1)
            for i in range(n)
        ]

        cancelled = threading.Event()
        errors: list[Exception] = []

        def fetch(idx: int, start: int, end: int) -> None:
            if cancelled.is_set():
                return
            try:
                headers = {"Range": f"bytes={start}-{end}"}
                with self.session.get(
                    file_url, headers=headers, stream=True, timeout=self.timeout
                ) as r:
                    if r.status_code != 206:
                        raise ValueError(
                            f"range request returned {r.status_code}, expected 206"
                        )
                    with open(parts[idx], "wb") as f:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if cancelled.is_set():
                                return
                            if should_cancel is not None and should_cancel():
                                cancelled.set()
                                return
                            if chunk:
                                f.write(chunk)
                                self.update_progress(len(chunk))
            except Exception as e:
                errors.append(e)
                cancelled.set()

        threads = [
            threading.Thread(target=fetch, args=(i, start, end))
            for i, start, end in bounds
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        def cleanup_parts() -> None:
            for p in parts:
                p.unlink(missing_ok=True)

        # Cancelled by the caller: drop everything and report cancellation.
        if should_cancel is not None and should_cancel():
            cleanup_parts()
            temp_file_path.unlink(missing_ok=True)
            return None

        # A range failed: undo the partial bytes we counted so the fallback's
        # own counting is not doubled, then retry the whole file on one
        # connection.
        if errors:
            partial = sum(p.stat().st_size for p in parts if p.exists())
            with self.lock:
                self.downloaded_bytes = max(0, self.downloaded_bytes - partial)
            cleanup_parts()
            temp_file_path.unlink(missing_ok=True)
            logger.warning(
                f"Parallel download failed ({errors[0]}); "
                f"falling back to a single connection"
            )
            return self._download_stream(file_url, temp_file_path, should_cancel)

        # Join the parts in order.
        try:
            with open(temp_file_path, "wb") as out:
                for p in parts:
                    with open(p, "rb") as pf:
                        shutil.copyfileobj(pf, out, length=1024 * 1024)
        finally:
            cleanup_parts()
        return file_size

    def download_repo(
        self,
        repo_id: str,
        local_dir: Path,
        progress_callback: Callable[[str, float], None] | None = None,
        revision: str = "main",
        should_cancel: Callable[[], bool] | None = None,
        include: set[str] | None = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Download a Hugging Face repository, or a named subset of it.

        Args:
            repo_id: Repository ID (e.g., "Addax-Data-Science/MDV5A")
            local_dir: Local directory to save files
            progress_callback: Optional callback(message, progress) for updates
            revision: Branch or revision to download
            should_cancel: Optional predicate polled while downloading; when
                it returns True the download is aborted and JobCancelledError
                is raised so the caller can clean up and report cancellation.
            include: Only download these repo-relative paths. None means the
                whole repo.
            overwrite: Fetch files even when a local file of the same size is
                already there.

        Returns:
            True if successful, False otherwise

        Raises:
            JobCancelledError: If should_cancel() returned True mid-download.
        """
        try:
            logger.info(f"Starting download of {repo_id} (revision: {revision})")

            if progress_callback:
                progress_callback(f"Fetching repository info for {repo_id}...", 0.0)

            # Get repository info and total size
            total_size, files_info = self.get_repo_info(repo_id, revision, include)
            self.total_bytes = total_size
            self.downloaded_bytes = 0
            self.start_time = time.time()

            size_gb = total_size / (1024 * 1024 * 1024)
            logger.info(f"Repository size: {size_gb:.2f} GB ({len(files_info)} files)")

            if progress_callback:
                progress_callback(
                    f"Downloading {len(files_info)} files ({size_gb:.2f} GB)...", 0.05
                )

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
                    executor.submit(
                        self.download_file, file_info, local_dir, should_cancel, overwrite
                    ): file_info
                    for file_info in files_info
                }

                # Send initial progress right after submitting tasks
                if progress_callback and total_size > 0:
                    progress_callback("Starting download...", 0.05)

                # Process completed downloads
                completed = 0

                while future_to_file:
                    # Cancel requested: stop scheduling new work and abort.
                    # In-flight download_file calls poll should_cancel too,
                    # so they unblock quickly; cancel_futures drops the rest.
                    if should_cancel is not None and should_cancel():
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise JobCancelledError()

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
                            logger.error(
                                f"Unexpected error downloading "
                                f"{file_info['path']}: {e}"
                            )
                            failed_downloads += 1

                        completed += 1

                    # Send periodic progress updates (even while downloading)
                    current_time = time.time()
                    time_since_last = current_time - last_progress_update
                    should_update = (
                        time_since_last >= progress_update_interval
                        or len(completed_futures) > 0  # Also update when files complete
                    )

                    # DEBUG: Log the check conditions
                    logger.debug(
                        f"Progress check: callback={progress_callback is not None}, "
                        f"total_size={total_size}, downloaded={self.downloaded_bytes}, "
                        f"time_since_last={time_since_last:.2f}s, "
                        f"completed_futures={len(completed_futures)}, "
                        f"should_update={should_update}"
                    )

                    if progress_callback and total_size > 0 and should_update:
                        last_progress_update = current_time
                        overall_progress = 0.05 + (self.downloaded_bytes / total_size) * 0.9

                        # Calculate download speed
                        elapsed = current_time - self.start_time
                        if elapsed > 0:
                            speed_mbps = (self.downloaded_bytes / elapsed) / (1024 * 1024)

                            # Only show download speed (progress bar already shows percentage)
                            logger.debug(
                                f"Sending progress: {overall_progress:.1%}, {speed_mbps:.2f} MB/s"
                            )
                            progress_callback(
                                f"Downloading model at {speed_mbps:.2f} MB/s",
                                overall_progress,
                            )

                    # Periodically adjust workers based on performance
                    self.adjust_workers()

                    # Short sleep to prevent busy waiting
                    time.sleep(0.1)

            # Summary
            logger.info(
                f"Download completed! Success: {successful_downloads}, Failed: {failed_downloads}"
            )

            if progress_callback:
                progress_callback("Download complete", 1.0)

            return failed_downloads == 0

        except JobCancelledError:
            # Cancelled mid-download; let the caller clean up the partial
            # directory and report cancellation rather than failure.
            logger.info(f"Download of {repo_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"Download failed: {e}", 0.0)
            return False
