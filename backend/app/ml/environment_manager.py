"""
Micromamba environment manager with static YAML-based environments.

Based on proven patterns from streamlit-AddaxAI.
Reads environment.yml files from backend/app/ml/envs/{env_name}/{platform}/.

Following DEVELOPERS.md principles:
- Crash early if setup fails
- Explicit error messages
- Type hints everywhere
"""

import os
import platform
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from app.core.logging_config import get_logger
from app.ml.schemas.model_manifest import ModelManifest
from app.utils.subprocess_runner import log_subprocess_failure, stream_with_tail

logger = get_logger(__name__)


class EnvironmentManager:
    """
    Manages micromamba environments using static YAML files.

    Environments are defined in backend/app/ml/envs/{env_name}/{platform}/environment.yml
    and created in ~/AddaxAI/envs/env-{env_name}/
    """

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
        self, manifest: ModelManifest, progress_callback: Callable[[str, float], None] | None = None
    ) -> Path:
        """
        Get existing environment or create new one from YAML.

        Args:
            manifest: Model manifest with env name
            progress_callback: Optional callback function(message: str, progress: float)

        Returns:
            Path to environment directory

        Raises:
            RuntimeError: If environment creation fails
            FileNotFoundError: If environment YAML not found
        """
        env_name = f"env-{manifest.env}"
        env_path = self.envs_dir / env_name

        # Check if environment exists and is valid
        if env_path.exists() and self._validate_env(env_path):
            logger.info(f"Using existing environment: {env_name}")
            # Don't call progress callback here - this should have been caught earlier
            # If we reach this point, it's likely a race condition or the validation
            # in the caller was incorrect. Just return the path silently.
            return env_path

        # Get environment YAML path
        yaml_path = self.get_env_yaml_path(manifest.env)

        # If env_path exists but is invalid (from failed previous attempt), remove it
        if env_path.exists():
            logger.warning(f"Removing invalid/incomplete environment at {env_path}")
            self._safe_rmtree(env_path)

        # Create new environment
        logger.info(f"Creating environment {env_name} from {yaml_path}")
        self._create_env(env_name, env_path, yaml_path, progress_callback)

        return env_path

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
    ) -> None:
        """
        Create micromamba environment from YAML file.

        Args:
            env_name: Environment name
            env_path: Path where environment will be created
            yaml_path: Path to environment.yml file
            progress_callback: Optional callback function(message: str, progress: float)

        Raises:
            RuntimeError: If environment creation fails
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
                    # If we can't remove it, try to continue anyway - micromamba might overwrite
                    logger.warning("Attempting to continue despite cleanup failure...")

            logger.info(f"Running micromamba create for {env_name} (temp: {temp_env_path})...")
            cmd = [
                str(self.micromamba_path),
                "create",
                "-f",
                str(yaml_path),
                "-p",
                str(temp_env_path),  # Create in temp location
                "-y",
                "-v",  # Verbose output for better progress tracking
                "--no-rc",  # Don't use .condarc
            ]

            # Subprocess env tuning. Verbose pip is needed for the line
            # parser below to surface progress. The retry / timeout knobs
            # mirror the legacy AddaxAI Windows workflow: a single dropped
            # TCP packet during the 2.3 GB torch download otherwise nukes
            # the whole install and the user has to start over.
            env = os.environ.copy()
            env["PIP_VERBOSE"] = "1"
            env["PIP_DEFAULT_TIMEOUT"] = "120"
            env["PIP_RETRIES"] = "5"
            env["MAMBA_REMOTE_CONNECT_TIMEOUT_SECS"] = "120"
            env["MAMBA_REMOTE_READ_TIMEOUT_SECS"] = "120"
            env["MAMBA_REMOTE_MAX_RETRIES"] = "5"

            current_progress = 0.0

            def on_micromamba_line(line: str) -> None:
                nonlocal current_progress
                # Simple checkpoint-based progress estimation
                if "using cache" in line.lower():
                    current_progress = max(current_progress, conda_start * 0.5)
                elif "transaction" in line.lower() and "starting" not in line.lower():
                    current_progress = max(current_progress, conda_start)
                elif line.startswith("Linking "):
                    current_progress = max(
                        current_progress,
                        conda_start + (conda_end - conda_start) * 0.5,
                    )
                elif "transaction finished" in line.lower():
                    current_progress = max(current_progress, conda_end)
                elif "installing pip packages" in line.lower():
                    current_progress = max(current_progress, pip_start)
                elif line.startswith("Collecting "):
                    pip_range = pip_end - pip_start
                    current_progress = max(
                        current_progress, pip_start + (pip_range * 0.3)
                    )
                elif "installing collected packages" in line.lower():
                    pip_range = pip_end - pip_start
                    current_progress = max(
                        current_progress, pip_start + (pip_range * 0.7)
                    )
                elif line.startswith("Successfully installed"):
                    current_progress = 0.95

                if progress_callback:
                    progress_callback(line[:80], current_progress)
                logger.debug(f"micromamba: {line}")

            result = stream_with_tail(
                cmd, env=env, on_line=on_micromamba_line
            )

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

            if progress_callback:
                progress_callback("Environment ready", 1.0)

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
        Validate that environment exists and has Python.

        Args:
            env_path: Path to environment

        Returns:
            True if valid, False otherwise
        """
        python_path = self._get_python_path(env_path)
        return python_path.exists()

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
