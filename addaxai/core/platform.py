"""Platform-specific utilities for AddaxAI.

OS detection, Python interpreter lookup, and DPI awareness.
"""

import os
import platform as _platform


def get_python_interpreter(base_path, env_name, system=None):
    """Return the path to the Python interpreter for a given conda environment.

    Args:
        base_path: Root AddaxAI directory containing the envs/ folder.
        env_name: Name of the environment (e.g. "base", "tensorflow").
        system: OS name override ("Windows", "Darwin", "Linux").
                Defaults to current platform.

    Returns:
        Absolute path to the python executable.
    """
    if system is None:
        system = _platform.system()

    if system == "Windows":
        return os.path.join(base_path, "envs", f"env-{env_name}", "python.exe")
    else:
        return os.path.join(base_path, "envs", f"env-{env_name}", "bin", "python")
