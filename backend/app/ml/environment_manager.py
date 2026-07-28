"""
Micromamba environment manager with static YAML-based environments.

Based on proven patterns from streamlit-AddaxAI.
Reads environment.yml files from backend/app/ml/envs/{env_name}/{platform}/.

Following DEVELOPERS.md principles:
- Crash early if setup fails
- Explicit error messages
- Type hints everywhere
"""

import hashlib
import os
import platform
import shutil
import subprocess
import threading
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from app.core.job_cancellation import JobCancelledError, is_cancel_requested
from app.core.logging_config import get_logger
from app.ml.schemas.model_manifest import ModelManifest
from app.utils.subprocess_env import clean_python_env
from app.utils.subprocess_runner import log_subprocess_failure, stream_with_tail

logger = get_logger(__name__)

# Hidden filename used to record which bundled YAML hash an env was
# built from. Lives inside the env directory so it gets removed
# automatically when the env is deleted via _safe_rmtree. Drift
# detection compares this to the current bundled YAML hash.
ENV_YAML_SHA_FILENAME = ".addaxai-yaml-sha256"


def hash_yaml_file(yaml_path: Path) -> str:
    """
    Full byte-level SHA-256 of an environment.yml. Comments and
    formatting count: any meaningful edit changes the bytes and we'd
    rather over-trigger a rebuild than miss one.
    """
    return hashlib.sha256(yaml_path.read_bytes()).hexdigest()


# Pre-conda baseline: matches the explicit "Starting package installation"
# preset emitted just before micromamba spawns. The bar must never visibly
# slide back to 0 while the resolve phase loads the package index, which
# can take 20-60 s on a cold cache or slow mirror.
ENV_PROGRESS_FLOOR = 0.10


def parse_micromamba_progress(
    line: str,
    current_progress: float,
    *,
    conda_start: float,
    conda_end: float,
    pip_start: float,
    pip_end: float,
) -> tuple[float, str]:
    """Map a line of micromamba verbose stderr to a (progress, caption).

    Progress is monotonically non-decreasing: every checkpoint uses
    max() against the caller's `current_progress`, so phases that
    re-emit older patterns never make the bar slide backwards.

    Caption is user-friendly when the line matches a known phase, and
    falls back to a truncated copy of the raw line otherwise (so the
    user always sees text changing under the bar, even on unrecognised
    phases). Returned alongside progress so callers don't have to
    re-match.

    Pattern order matters: more specific patterns come first. Phase
    progressions are roughly:
      resolve (1-5%) -> conda download/link (5%-conda_end) -> pip (conda_end-95%).
    """
    pip_range = pip_end - pip_start
    lower = line.lower()

    # Resolve phase. Each step is cheap individually but the sum can
    # easily stretch past a minute on cold caches. We move the bar by
    # ~1% at each, so the user sees forward motion even before any
    # actual package download has begun.
    if "searching index cache" in lower:
        return max(current_progress, 0.01), "Loading package index..."
    if "fetch shard index" in lower:
        return max(current_progress, 0.02), "Loading package index..."
    if "parsing packages" in lower:
        return max(current_progress, 0.03), "Resolving dependencies..."
    if "resolving environment" in lower:
        return max(current_progress, 0.04), "Resolving dependencies..."

    # Conda download / link phase.
    if "transaction" in lower and "starting" not in lower:
        return max(current_progress, conda_start), "Downloading packages..."
    if "using cache" in lower:
        # Older event: package weights were already cached, conda is
        # reusing them. Tiny lift between resolve and transaction.
        return max(current_progress, conda_start * 0.5), "Downloading packages..."
    if line.startswith("Linking "):
        midpoint = conda_start + (conda_end - conda_start) * 0.5
        return max(current_progress, midpoint), "Installing packages..."
    if "transaction finished" in lower:
        return max(current_progress, conda_end), "Conda packages installed"

    # Pip phase.
    if "installing pip packages" in lower:
        return max(current_progress, pip_start), "Installing Python packages..."
    if line.startswith("Collecting "):
        return (
            max(current_progress, pip_start + pip_range * 0.3),
            "Downloading Python packages...",
        )
    if "installing collected packages" in lower:
        return (
            max(current_progress, pip_start + pip_range * 0.7),
            "Installing Python packages...",
        )
    if line.startswith("Successfully installed"):
        return max(current_progress, 0.95), "Python packages installed"

    # Byte-compiling installed packages: the last conda link step. It goes
    # silent for a stretch with no per-file output, so without this the bar
    # freezes on a raw "libmamba Waiting for pyc compilation" line and looks
    # stuck (the beta report). We pass --no-pyc so this normally does not run,
    # but keep the caption as a safety net for builds / pip that still compile.
    if "pyc" in lower:
        finishing = conda_start + (conda_end - conda_start) * 0.85
        return (
            max(current_progress, finishing),
            "Finishing install, compiling files (can take a few minutes)...",
        )

    # No known phase: leave progress alone, show the raw line so the
    # user still sees activity. Truncated so a 500-char libmamba diag
    # line doesn't blow up the dialog.
    return current_progress, line[:80]


class EnvironmentManager:
    """
    Manages micromamba environments using static YAML files.

    Environments are defined in backend/app/ml/envs/{env_name}/{platform}/environment.yml
    and created in ~/AddaxAI/envs/env-{env_name}/
    """

    # Class-level so the locks are shared across every `EnvironmentManager`
    # instance in the process. Two callers (e.g. the setup install-env
    # path and the per-model prepare-env path) used to race on
    # `.<env>.tmp/` because they constructed their own managers; now they
    # serialise on the same per-env lock no matter who owns the instance.
    _env_locks: ClassVar[dict[str, threading.Lock]] = {}
    _env_locks_registry_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _env_build_lock(cls, env_name: str) -> threading.Lock:
        """Return the (lazily-created) lock that gates builds of `env_name`."""
        with cls._env_locks_registry_lock:
            lock = cls._env_locks.get(env_name)
            if lock is None:
                lock = threading.Lock()
                cls._env_locks[env_name] = lock
            return lock

    def __init__(self, envs_dir: Path | None = None, micromamba_path: Path | None = None):
        """
        Initialize environment manager.

        Args:
            envs_dir: Directory to store environments (default: ~/AddaxAI/envs)
            micromamba_path: Path to micromamba binary (default: ~/AddaxAI/bin/micromamba)
        """
        user_data_dir = Path.home() / "AddaxAI"
        self.envs_dir = envs_dir or (user_data_dir / "envs")

        bin_dir = user_data_dir / "bin"
        # Windows uses .exe extension
        micromamba_name = "micromamba.exe" if platform.system() == "Windows" else "micromamba"
        self.micromamba_path = micromamba_path or (bin_dir / micromamba_name)

        self._ensure_runtime_dirs()

    def _ensure_runtime_dirs(self) -> None:
        """
        Make sure the on-disk state this manager depends on actually exists.
        Called at construction time and again before any micromamba invocation
        so the manager self-heals if `~/AddaxAI/bin` or `~/AddaxAI/envs` got
        wiped underneath us (Reset application, antivirus quarantine, manual
        rm). Without this we hit ENOENT inside subprocess.Popen and there is
        no recovery without a server restart.
        """
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.micromamba_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.micromamba_path.exists():
            logger.info("Micromamba not found, downloading...")
            self._download_micromamba()

    def _download_micromamba(self):
        """Download micromamba binary for the current platform."""
        system = platform.system()
        machine = platform.machine()

        # Determine download URL based on platform
        if system == "Darwin":
            if machine == "arm64":
                url = "https://micro.mamba.pm/api/micromamba/osx-arm64/latest"
            else:
                url = "https://micro.mamba.pm/api/micromamba/osx-64/latest"
        elif system == "Linux":
            if machine == "aarch64":
                url = "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest"
            else:
                url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
        elif system == "Windows":
            url = "https://micro.mamba.pm/api/micromamba/win-64/latest"
        else:
            raise RuntimeError(
                f"Unsupported platform: {system} {machine}. " f"Please install micromamba manually."
            )

        logger.info(f"Downloading micromamba from {url}")

        try:
            # Download the compressed tar archive
            with urllib.request.urlopen(url, timeout=60) as response:
                compressed_content = response.read()

            logger.info(f"Downloaded {len(compressed_content)} bytes")

            # Decompress bz2
            import bz2
            import tarfile
            import tempfile

            tar_content = bz2.decompress(compressed_content)
            logger.info(f"Decompressed to {len(tar_content)} bytes")

            # Extract bin/micromamba from tar archive
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp_tar:
                tmp_tar.write(tar_content)
                tmp_tar_path = tmp_tar.name

            try:
                with tarfile.open(tmp_tar_path, "r") as tar:
                    # Windows uses Library/bin/micromamba.exe, others use bin/micromamba
                    if system == "Windows":
                        member_path = "Library/bin/micromamba.exe"
                    else:
                        member_path = "bin/micromamba"
                    member = tar.getmember(member_path)
                    member_file = tar.extractfile(member)
                    if member_file:
                        with open(self.micromamba_path, "wb") as f:
                            f.write(member_file.read())
                        logger.info(f"Extracted micromamba binary from {member_path}")
            finally:
                Path(tmp_tar_path).unlink()

            # Make executable
            self.micromamba_path.chmod(0o755)
            logger.info(f"Micromamba installed successfully at {self.micromamba_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to download micromamba: {e}") from e

    def get_env_yaml_path(self, env_name: str) -> Path:
        """
        Get path to environment YAML file for current platform.

        Args:
            env_name: Environment name (e.g., "megadetector", "pytorch-classifier")

        Returns:
            Path to environment.yml file

        Raises:
            FileNotFoundError: If environment YAML not found
        """
        # Determine platform directory
        system = platform.system().lower()
        if system == "darwin":
            platform_dir = "darwin"
        elif system == "linux":
            platform_dir = "linux"
        elif system == "windows":
            platform_dir = "windows"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        # Path to YAML file in repo
        # backend/app/ml/envs/{env_name}/{platform}/environment.yml
        backend_root = Path(__file__).parent
        yaml_path = backend_root / "envs" / env_name / platform_dir / "environment.yml"

        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Environment YAML not found: {yaml_path}\n"
                f"Expected location: backend/app/ml/envs/{env_name}/{platform_dir}/environment.yml"
            )

        return yaml_path

    def get_or_create_env(
        self,
        manifest: ModelManifest,
        progress_callback: Callable[[str, float], None] | None = None,
        job_id: str | None = None,
    ) -> Path:
        """
        Get existing environment or create new one from YAML.

        Args:
            manifest: Model manifest with env name
            progress_callback: Optional callback function(message: str, progress: float)
            job_id: Optional cancellation key. When set, the micromamba
                subprocess is killable via `request_cancel(job_id)` and a
                cancel mid-build raises `JobCancelledError`. None for the
                non-cancellable callers (first-run setup, drift rebuild).

        Returns:
            Path to environment directory

        Raises:
            RuntimeError: If environment creation fails
            JobCancelledError: If the build was cancelled via job_id
            FileNotFoundError: If environment YAML not found
        """
        env_name = f"env-{manifest.env}"
        env_path = self.envs_dir / env_name

        # Cheap fast-path before we touch the lock: if the env is
        # already complete, no build is needed regardless of who else
        # might be working on a different env.
        if env_path.exists() and self._validate_env(env_path):
            logger.info(f"Using existing environment: {env_name}")
            return env_path

        # Serialise concurrent builds of the same env. Two pathways
        # (`/api/setup/install-env` and `/api/ml/models/{id}/prepare-env`)
        # used to race on `.{env_name}.tmp/` because they each created
        # their own `EnvironmentManager`; the lock is class-level so it
        # gates them both. Another env (different name) can still build
        # in parallel because each name has its own lock.
        lock = self._env_build_lock(env_name)
        already_held = not lock.acquire(blocking=False)
        if already_held:
            logger.info(
                f"Another build of {env_name} is in flight; waiting for "
                "it to finish before proceeding"
            )
            if progress_callback:
                progress_callback(
                    f"Another rebuild of {env_name} is already running; "
                    "waiting for it to finish...",
                    0.0,
                )
            lock.acquire(blocking=True)

        try:
            # The previous holder may have just finished a successful
            # build; pick up its result instead of starting our own.
            if env_path.exists() and self._validate_env(env_path):
                logger.info(
                    f"{env_name} was built while we were waiting; "
                    "skipping our own build"
                )
                return env_path

            yaml_path = self.get_env_yaml_path(manifest.env)

            # If env_path exists but is invalid (from failed previous
            # attempt), remove it before retrying.
            if env_path.exists():
                logger.warning(
                    f"Removing invalid/incomplete environment at {env_path}"
                )
                self._safe_rmtree(env_path)

            logger.info(f"Creating environment {env_name} from {yaml_path}")
            self._create_env(env_name, env_path, yaml_path, progress_callback, job_id)

            return env_path
        finally:
            lock.release()

    def _parse_env_yaml(self, yaml_path: Path) -> tuple[int, int]:
        """
        Parse environment.yml to count conda and pip packages.

        Args:
            yaml_path: Path to environment.yml file

        Returns:
            Tuple of (conda_package_count, pip_package_count)
        """
        import yaml

        with open(yaml_path) as f:
            env_config = yaml.safe_load(f)

        conda_count = 0
        pip_count = 0

        # Count conda packages (top-level dependencies, excluding pip itself)
        dependencies = env_config.get("dependencies", [])
        for dep in dependencies:
            if isinstance(dep, str):
                # Skip 'pip' entry itself
                if not dep.startswith("pip"):
                    conda_count += 1
            elif isinstance(dep, dict) and "pip" in dep:
                # Count pip packages
                pip_packages = dep["pip"]
                for pkg in pip_packages:
                    # Skip comments and flags
                    if isinstance(pkg, str) and not pkg.strip().startswith("#"):
                        pip_count += 1

        return conda_count, pip_count

    def _create_env(
        self,
        env_name: str,
        env_path: Path,
        yaml_path: Path,
        progress_callback: Callable[[str, float], None] | None = None,
        job_id: str | None = None,
    ) -> None:
        """
        Create micromamba environment from YAML file.

        Args:
            env_name: Environment name
            env_path: Path where environment will be created
            yaml_path: Path to environment.yml file
            progress_callback: Optional callback function(message: str, progress: float)
            job_id: Optional cancellation key passed to stream_with_tail so
                the micromamba subprocess can be killed on cancel.

        Raises:
            RuntimeError: If environment creation fails
            JobCancelledError: If the build was cancelled via job_id
        """
        try:
            # Heal any missing on-disk state (bin/, envs/, micromamba binary)
            # before invoking subprocess. Reset application + a stale cached
            # manager would otherwise crash with ENOENT inside Popen.
            self._ensure_runtime_dirs()

            # Parse YAML to count packages and allocate progress ranges
            conda_count, pip_count = self._parse_env_yaml(yaml_path)
            logger.info(
                f"Environment has {conda_count} conda packages and {pip_count} pip packages"
            )

            # Dynamically allocate progress based on package counts
            # Each package (conda or pip) is weighted equally
            total_packages = conda_count + pip_count

            if total_packages > 0:
                conda_progress_range = conda_count / total_packages
                pip_count / total_packages
            else:
                # Fallback if no packages (shouldn't happen)
                conda_progress_range = 0.5

            # Reserve 5% for initial setup, distribute rest between conda and pip
            conda_start = 0.05
            conda_end = 0.05 + (0.95 * conda_progress_range)
            pip_start = conda_end
            pip_end = 1.0

            logger.info(
                f"Progress allocation: conda {conda_start:.0%}-{conda_end:.0%}, "
                f"pip {pip_start:.0%}-{pip_end:.0%}"
            )
            # Create environment with micromamba
            if progress_callback:
                progress_callback("Starting package installation...", 0.1)

            # Create environment in temporary location first for atomic operation
            temp_env_path = env_path.parent / f".{env_name}.tmp"

            # Clean up any existing temp directory from previous failed attempts
            if temp_env_path.exists():
                logger.warning(f"Removing stale temporary environment at {temp_env_path}")
                try:
                    self._safe_rmtree(temp_env_path)
                except Exception as e:
                    logger.error(f"Failed to remove stale temp directory: {e}")
                    # On Windows a killed run can leave the temp dir locked
                    # (open handle, antivirus scan). Building into a leftover
                    # half-env is exactly what made a retry stall early, so
                    # don't reuse it: fall back to a fresh unique temp path.
                    temp_env_path = (
                        env_path.parent / f".{env_name}.tmp-{uuid.uuid4().hex[:8]}"
                    )
                    logger.warning(
                        f"Building into a fresh temp path instead: {temp_env_path}"
                    )

            # micromamba materialises temp files next to the spec file
            # (pip requirement fragments), so the yaml's directory must be
            # writable. The bundled yaml lives inside the installed app
            # resources, which are read-only on Linux (AppImage FUSE mount,
            # root-owned /opt for a deb). Build from a copy in the writable
            # envs dir instead.
            yaml_copy_path = env_path.parent / f".{env_name}.environment.yml"
            shutil.copyfile(yaml_path, yaml_copy_path)

            logger.info(f"Running micromamba create for {env_name} (temp: {temp_env_path})...")
            cmd = [
                str(self.micromamba_path),
                "create",
                "-f",
                str(yaml_copy_path),
                "-p",
                str(temp_env_path),  # Create in temp location
                "-y",
                "-v",  # Verbose output for better progress tracking
                "--no-rc",  # Don't use .condarc
                # Skip byte-compiling .py to .pyc during the conda link step.
                # That step goes silent for minutes (worse on corporate
                # Windows where antivirus scans each of the thousands of new
                # files), which looked like a hang to a beta tester. Python
                # compiles each module on first import instead: a tiny,
                # one-time cost, not worth a multi-minute stall at install.
                "--no-pyc",
                # Fail after waiting 5 min for a package lock instead of
                # hanging forever on a stale lock left by a killed run.
                "--lock-timeout",
                "300",
            ]

            # Subprocess env tuning. Verbose pip is needed for the line
            # parser below to surface progress. The retry / timeout knobs
            # mirror the legacy AddaxAI Windows workflow: a single dropped
            # TCP packet during the 2.3 GB torch download otherwise nukes
            # the whole install and the user has to start over.
            # clean_python_env keeps the user's personal site-packages and
            # PYTHONPATH out of the build (pip runs inside the new env).
            env = clean_python_env()
            env["PIP_VERBOSE"] = "1"
            env["PIP_DEFAULT_TIMEOUT"] = "120"
            env["PIP_RETRIES"] = "5"
            env["MAMBA_REMOTE_CONNECT_TIMEOUT_SECS"] = "120"
            env["MAMBA_REMOTE_READ_TIMEOUT_SECS"] = "120"
            env["MAMBA_REMOTE_MAX_RETRIES"] = "5"

            # pip stages downloads and build dirs in TMPDIR. On Ubuntu
            # 24.10+ /tmp is a tmpfs capped at half the RAM with per-user
            # quotas, and the multi-GB torch download dies there with
            # EDQUOT (Linux beta report 2026-07-05). Point pip at a
            # disk-backed dir we own instead. POSIX only: Windows uses
            # TEMP/TMP and has no tmpfs problem, so leave it untouched.
            pip_tmp_dir = env_path.parent / f".{env_name}.pip-tmp"
            if os.name == "posix":
                pip_tmp_dir.mkdir(parents=True, exist_ok=True)
                env["TMPDIR"] = str(pip_tmp_dir)

            # Seed at the floor we just announced via the explicit
            # "Starting package installation..." callback above. The
            # first uncategorised micromamba line would otherwise emit
            # 0.0 and the bar would slide back to zero for the entire
            # ~30 s resolve phase.
            current_progress = ENV_PROGRESS_FLOOR

            def on_micromamba_line(line: str) -> None:
                nonlocal current_progress
                new_progress, caption = parse_micromamba_progress(
                    line,
                    current_progress,
                    conda_start=conda_start,
                    conda_end=conda_end,
                    pip_start=pip_start,
                    pip_end=pip_end,
                )
                current_progress = new_progress
                if progress_callback:
                    progress_callback(caption, current_progress)
                logger.debug(f"micromamba: {line}")

            try:
                result = stream_with_tail(
                    cmd, env=env, on_line=on_micromamba_line, job_id=job_id
                )
            finally:
                yaml_copy_path.unlink(missing_ok=True)
                shutil.rmtree(pip_tmp_dir, ignore_errors=True)

            # A cancel kills micromamba mid-run, which surfaces here as a
            # non-zero exit. Distinguish that from a genuine build failure
            # so the user sees "cancelled", not a scary error. Clean the
            # half-built temp env either way.
            if job_id is not None and is_cancel_requested(job_id):
                if temp_env_path.exists():
                    self._safe_rmtree(temp_env_path)
                logger.info(f"Environment build for {env_name} cancelled")
                raise JobCancelledError()

            if result.returncode != 0:
                # Clean up failed temp environment
                if temp_env_path.exists():
                    logger.warning(f"Removing failed temporary environment at {temp_env_path}")
                    self._safe_rmtree(temp_env_path)
                # Surface the captured tail at ERROR so backend.log holds
                # the pip stack-trace, not just the libmamba summary line.
                log_subprocess_failure("micromamba create", cmd, result)
                raise RuntimeError(
                    f"micromamba create failed:\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Last output: {result.last_line}"
                )

            if progress_callback:
                progress_callback("Packages installed successfully", 0.95)

            logger.info(f"Environment created in temporary location: {temp_env_path}")

            # Atomic rename - only move to final location if creation was successful
            if progress_callback:
                progress_callback("Finalizing environment...", 0.98)

            logger.info(f"Moving environment from {temp_env_path} to {env_path}")
            temp_env_path.rename(env_path)

            logger.info(f"Environment {env_name} created successfully at {env_path}")

            # Record which bundled YAML this env was built from so a
            # later drift check can spot when the YAML moves on. Best
            # effort: a write failure here doesn't break the install,
            # it just means drift detection won't fire for this env
            # until the next successful create.
            try:
                sentinel = env_path / ENV_YAML_SHA_FILENAME
                sentinel.write_text(hash_yaml_file(yaml_path))
                logger.info(
                    f"Wrote YAML hash sentinel to {sentinel}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to write YAML hash sentinel for {env_name}: {e}"
                )

            if progress_callback:
                progress_callback("Environment ready", 1.0)

        except JobCancelledError:
            # Cancelled mid-build. The temp env was already removed at the
            # cancel checkpoint above; propagate cleanly so the worker
            # reports cancellation rather than a build failure.
            raise
        except Exception as e:
            # Clean up failed environment - only if rename hasn't happened yet
            # If temp still exists, remove it. If rename happened, remove final location.
            if temp_env_path.exists():
                logger.warning(f"Cleaning up failed temporary environment at {temp_env_path}")
                try:
                    self._safe_rmtree(temp_env_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp environment: {cleanup_error}")
            elif env_path.exists():
                # Rename happened but something failed after
                logger.warning(f"Cleaning up failed environment at {env_path}")
                try:
                    self._safe_rmtree(env_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up environment: {cleanup_error}")
            raise RuntimeError(f"Failed to create environment {env_name}: {e}") from e

    def _validate_env(self, env_path: Path) -> bool:
        """
        Validate that an environment exists, has its Python binary on
        disk, AND that interpreter actually boots. The boot probe
        catches envs whose `python.exe` file survived but whose stdlib
        was pruned — Windows Defender quarantining `Lib/encodings/`,
        antivirus / Storage Sense cleanup, partial copy, interrupted
        rename, etc. Without this probe, the env reports "valid" right
        up until a worker subprocess crashes with
        `ModuleNotFoundError: No module named 'encodings'` and the user
        sees only "Classification worker exited with code 1".

        Returns False on any failure (missing binary, non-zero exit,
        timeout, OSError) so the caller treats the env as broken and
        triggers a rebuild.
        """
        python_path = self._get_python_path(env_path)
        if not python_path.exists():
            return False

        try:
            result = subprocess.run(
                [str(python_path), "-c", "import sys; sys.exit(0)"],
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(
                f"Boot probe failed for env at {env_path}: {e}"
            )
            return False

        if result.returncode != 0:
            stderr_tail = (result.stderr or b"").decode(errors="replace")[-500:]
            logger.warning(
                f"Boot probe at {env_path} exited with "
                f"{result.returncode}; stderr tail: {stderr_tail}"
            )
            return False

        return True

    def check_yaml_drift(self, env_name: str) -> bool | None:
        """
        Check whether an installed environment's YAML hash sentinel
        still matches the current bundled YAML.

        Returns:
            True  if the sentinel is present and disagrees with the
                  current bundled YAML hash (drift; rebuild needed).
            False if the sentinel is present and agrees (in sync).
            None  if the env or sentinel is absent (legacy install
                  predating drift detection, or env not built yet).
                  Caller should treat as "unknown but valid" and skip.

        Never raises: filesystem errors, missing YAML files, and
        unsupported platforms all collapse into None so a drift check
        can't take down startup.
        """
        env_path = self.envs_dir / f"env-{env_name}"
        if not env_path.exists():
            return None

        sentinel = env_path / ENV_YAML_SHA_FILENAME
        if not sentinel.exists():
            return None

        try:
            stored = sentinel.read_text().strip()
            yaml_path = self.get_env_yaml_path(env_name)
            current = hash_yaml_file(yaml_path)
        except (OSError, FileNotFoundError, RuntimeError) as e:
            logger.warning(
                f"Could not check YAML drift for env {env_name}: {e}"
            )
            return None

        return stored != current

    def _safe_rmtree(self, path: Path) -> None:
        """
        Safely remove a directory tree, handling permission errors on macOS.

        Args:
            path: Path to directory to remove
        """
        import stat

        def handle_remove_readonly(func, path, exc):
            """Handle permission errors by making files writable."""
            if not os.access(path, os.W_OK):
                # Change permissions and retry
                os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
                func(path)
            else:
                raise

        shutil.rmtree(path, onerror=handle_remove_readonly)

    def get_python(self, env_name: str) -> Path:
        """
        Get path to Python executable in environment.

        Args:
            env_name: Environment name (with env- prefix, e.g., "env-megadetector")

        Returns:
            Path to python executable

        Raises:
            FileNotFoundError: If environment doesn't exist
        """
        env_path = self.envs_dir / env_name
        if not env_path.exists():
            raise FileNotFoundError(f"Environment not found: {env_name}")

        python_path = self._get_python_path(env_path)
        if not python_path.exists():
            raise FileNotFoundError(f"Python not found in environment {env_name}: {python_path}")

        return python_path

    def _get_python_path(self, env_path: Path) -> Path:
        """Get Python executable path for platform."""
        if platform.system() == "Windows":
            return env_path / "python.exe"
        return env_path / "bin" / "python"
