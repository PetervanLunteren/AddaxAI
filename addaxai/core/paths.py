"""Path resolution for AddaxAI.

Provides functions to derive standard directory paths from the base install path.
"""

import os


def get_base_path(script_file: str) -> str:
    """Derive the AddaxAI base path from a script file location.

    Assumes the script is at base/AddaxAI/script.py, so base is two levels up.

    Args:
        script_file: Absolute path to the calling script file.

    Returns:
        Absolute path to the AddaxAI root directory.
    """
    return os.path.dirname(os.path.dirname(os.path.realpath(script_file)))


def get_cls_dir(base_path: str) -> str:
    """Return the classifier models directory."""
    return os.path.join(base_path, "models", "cls")


def get_det_dir(base_path: str) -> str:
    """Return the detector models directory."""
    return os.path.join(base_path, "models", "det")


def get_env_dir(base_path: str) -> str:
    """Return the conda environments directory."""
    return os.path.join(base_path, "envs")


def get_version(base_path: str) -> str:
    """Read and return the version string from AddaxAI/version.txt."""
    version_file = os.path.join(base_path, "AddaxAI", "version.txt")
    with open(version_file, 'r') as f:
        return f.read().strip()
