"""
Micromamba environment manager with content-based caching.

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

from app.core.logging_config import get_logger
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)


class EnvironmentManager:
    """
    Manages micromamba environments with content-based caching.

    Environments are cached based on content hash of their dependencies.
    If dependencies change, old environment is removed and new one is created.
    This ensures environments are always up-to-date while being fast to reuse.
    """

    def __init__(self, envs_dir: Path | None = None, micromamba_dir: Path | None = None):
        """
        Initialize environment manager.

        Args:
            envs_dir: Directory to store environments (default: ~/AddaxAI/ml_envs)
            micromamba_dir: Directory for micromamba binary (default: ~/AddaxAI/bin)
        """
        user_data_dir = Path.home() / "AddaxAI"
        self.envs_dir = envs_dir or (user_data_dir / "ml_envs")
        self.micromamba_dir = micromamba_dir or (user_data_dir / "bin")

        # Create directories
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        self.micromamba_dir.mkdir(parents=True, exist_ok=True)

        # Get or download micromamba
        self.micromamba_path = self._get_micromamba()

    def _get_micromamba(self) -> Path:
        """
        Get path to micromamba executable (download if needed).

        Returns:
            Path to micromamba executable

        Raises:
            RuntimeError: If download/setup fails
        """
        # Check if micromamba already exists
        if platform.system() == "Windows":
            exe_name = "micromamba.exe"
        else:
            exe_name = "micromamba"

        micromamba_path = self.micromamba_dir / exe_name

        if micromamba_path.exists():
            logger.info(f"Using existing micromamba: {micromamba_path}")
            return micromamba_path

        # Download micromamba
        logger.info("Downloading micromamba...")
        download_url = self._get_micromamba_download_url()

        try:
            with urllib.request.urlopen(download_url, timeout=60) as response:
                with open(micromamba_path, "wb") as f:
                    f.write(response.read())

            # Make executable on Unix
            if platform.system() != "Windows":
                micromamba_path.chmod(0o755)

            logger.info(f"Downloaded micromamba to {micromamba_path}")
            return micromamba_path

        except Exception as e:
            raise RuntimeError(
                f"Failed to download micromamba from {download_url}: {e}"
            ) from e

    def _get_micromamba_download_url(self) -> str:
        """
        Get micromamba download URL for current platform.

        Returns:
            Download URL

        Raises:
            RuntimeError: If platform is not supported
        """
        system = platform.system()
        machine = platform.machine().lower()

        base_url = "https://micro.mamba.pm/api/micromamba"

        if system == "Linux":
            if machine in ("x86_64", "amd64"):
                return f"{base_url}/linux-64/latest"
            elif machine in ("aarch64", "arm64"):
                return f"{base_url}/linux-aarch64/latest"
        elif system == "Darwin":  # macOS
            if machine in ("x86_64", "amd64"):
                return f"{base_url}/osx-64/latest"
            elif machine in ("arm64", "aarch64"):
                return f"{base_url}/osx-arm64/latest"
        elif system == "Windows":
            return f"{base_url}/win-64/latest"

        raise RuntimeError(
            f"Unsupported platform: {system} {machine}. "
            f"Please install micromamba manually."
        )

    def get_or_create_env(self, manifest: ModelManifest) -> Path:
        """
        Get cached environment or create new one.

        Uses content-based caching: if dependencies haven't changed,
        reuses existing environment. If changed, creates new one.

        Args:
            manifest: Model manifest with environment specifications

        Returns:
            Path to environment directory

        Raises:
            RuntimeError: If environment creation fails
        """
        # Generate environment YAML
        env_yaml = self._generate_env_yaml(manifest)

        # Hash the env.yaml content
        env_hash = hashlib.sha256(env_yaml.encode()).hexdigest()[:8]

        # Environment name: {base_name}_{hash}
        env_name = f"{manifest.environment}_{env_hash}"
        env_path = self.envs_dir / env_name

        # Check if environment exists and is valid
        if env_path.exists() and self._validate_env(env_path):
            logger.info(f"Using cached environment: {env_name}")
            return env_path

        # Clean old environments with same base name
        self._clean_old_envs(manifest.environment, keep=env_name)

        # Create new environment
        logger.info(f"Creating new environment: {env_name} (hash: {env_hash})")
        self._create_env(env_name, env_path, manifest)

        return env_path

    def _generate_env_yaml(self, manifest: ModelManifest) -> str:
        """
        Generate environment.yaml content from manifest.

        Args:
            manifest: Model manifest

        Returns:
            YAML content as string
        """
        # Convert dependencies list to YAML format
        pip_deps = "\n    - ".join(manifest.dependencies)

        yaml_content = f"""name: {manifest.environment}
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python={manifest.python_version}
  - pip
  - pip:
    - {pip_deps}
"""
        return yaml_content

    def _create_env(self, env_name: str, env_path: Path, manifest: ModelManifest) -> None:
        """
        Create micromamba environment.

        Args:
            env_name: Environment name
            env_path: Path where environment will be created
            manifest: Model manifest

        Raises:
            RuntimeError: If environment creation fails
        """
        # Write env.yaml to temp file
        env_yaml = self._generate_env_yaml(manifest)
        tmp_yaml = self.envs_dir / f"{env_name}.yaml"
        tmp_yaml.write_text(env_yaml)

        try:
            # Create environment with micromamba
            logger.info(f"Running micromamba create for {env_name}...")
            cmd = [
                str(self.micromamba_path),
                "create",
                "-f", str(tmp_yaml),
                "-p", str(env_path),
                "-y",
                "--no-rc",  # Don't use .condarc
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"micromamba create failed:\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stdout: {result.stdout}\n"
                    f"Stderr: {result.stderr}"
                )

            logger.info(f"Environment {env_name} created successfully")

            # GPU fix for Windows (PyTorch CUDA)
            if platform.system() == "Windows" and "torch" in manifest.dependencies[0].lower():
                logger.info("Applying Windows GPU fix (reinstalling PyTorch with CUDA)...")
                python = self.get_python(env_name)
                gpu_fix_cmd = [
                    str(python), "-m", "pip", "install",
                    "torch", "torchvision",
                    "--upgrade", "--force-reinstall",
                    "--index-url", "https://download.pytorch.org/whl/cu118"
                ]
                subprocess.run(gpu_fix_cmd, check=True, capture_output=True)
                logger.info("GPU fix applied")

        except Exception as e:
            # Clean up failed environment
            if env_path.exists():
                shutil.rmtree(env_path)
            raise RuntimeError(f"Failed to create environment {env_name}: {e}") from e
        finally:
            # Clean up temp YAML
            if tmp_yaml.exists():
                tmp_yaml.unlink()

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

    def _clean_old_envs(self, env_base: str, keep: str) -> None:
        """
        Delete old environment versions with same base name.

        Args:
            env_base: Base environment name (e.g., "megadetector-env")
            keep: Full environment name to keep (e.g., "megadetector-env_abc123")
        """
        for env_dir in self.envs_dir.glob(f"{env_base}_*"):
            if env_dir.name != keep:
                logger.info(f"Removing old environment: {env_dir.name}")
                try:
                    shutil.rmtree(env_dir)
                except Exception as e:
                    logger.warning(f"Failed to remove old environment {env_dir.name}: {e}")

    def get_python(self, env_name: str) -> Path:
        """
        Get path to Python executable in environment.

        Args:
            env_name: Environment name (with hash)

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
            raise FileNotFoundError(
                f"Python not found in environment {env_name}: {python_path}"
            )

        return python_path

    def _get_python_path(self, env_path: Path) -> Path:
        """Get Python executable path for platform."""
        if platform.system() == "Windows":
            return env_path / "python.exe"
        return env_path / "bin" / "python"

    def list_environments(self) -> list[str]:
        """
        List all installed environments.

        Returns:
            List of environment names
        """
        if not self.envs_dir.exists():
            return []

        return [d.name for d in self.envs_dir.iterdir() if d.is_dir()]

    def remove_environment(self, env_name: str) -> bool:
        """
        Remove an environment.

        Args:
            env_name: Environment name to remove

        Returns:
            True if removed, False if didn't exist
        """
        env_path = self.envs_dir / env_name
        if not env_path.exists():
            return False

        logger.info(f"Removing environment: {env_name}")
        shutil.rmtree(env_path)
        return True
