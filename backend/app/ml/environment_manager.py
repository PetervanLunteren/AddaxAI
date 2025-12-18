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
import urllib.request
from pathlib import Path
from typing import Callable

from app.core.logging_config import get_logger
from app.ml.schemas.model_manifest import ModelManifest

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
        self.envs_dir.mkdir(parents=True, exist_ok=True)

        bin_dir = user_data_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        self.micromamba_path = micromamba_path or (bin_dir / "micromamba")

        # Ensure micromamba is installed
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
                f"Unsupported platform: {system} {machine}. "
                f"Please install micromamba manually."
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
                    member = tar.getmember("bin/micromamba")
                    member_file = tar.extractfile(member)
                    if member_file:
                        with open(self.micromamba_path, "wb") as f:
                            f.write(member_file.read())
                        logger.info("Extracted micromamba binary from tar archive")
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
            if progress_callback:
                progress_callback(f"Environment {env_name} already exists", 1.0)
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
            # Create environment with micromamba
            if progress_callback:
                progress_callback("Starting package installation...", 0.1)

            logger.info(f"Running micromamba create for {env_name}...")
            cmd = [
                str(self.micromamba_path),
                "create",
                "-f",
                str(yaml_path),
                "-p",
                str(env_path),
                "-y",
                "--no-rc",  # Don't use .condarc
            ]

            # Stream output line by line to show progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            last_line = ""
            for line in process.stdout:
                line = line.strip()
                if line:
                    last_line = line
                    # Send progress updates with the current line
                    if progress_callback:
                        progress_callback(line[:80], 0.5)
                    logger.debug(f"micromamba: {line}")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(
                    f"micromamba create failed:\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Last output: {last_line}"
                )

            if progress_callback:
                progress_callback("Packages installed successfully", 0.8)

            logger.info(f"Environment {env_name} created successfully")

            if progress_callback:
                progress_callback("Environment ready", 1.0)

        except Exception as e:
            # Clean up failed environment
            if env_path.exists():
                logger.warning(f"Cleaning up failed environment at {env_path}")
                self._safe_rmtree(env_path)
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
