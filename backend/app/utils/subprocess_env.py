"""Clean environment for spawned Python subprocesses.

AddaxAI's ML envs are micromamba envs, not venvs, so their pythons keep
user-site enabled: anything the user ever ``pip install --user``-ed for a
matching Python minor version lands on ``sys.path`` BEFORE the env's own
site-packages and shadows the pinned packages (the classic "conda env
broken by ~/.local" failure). A globally exported ``PYTHONPATH`` (common
on Anaconda setups) leaks the same way with higher precedence.

Every spawn of an env python or micromamba must therefore go through
``clean_python_env()`` instead of inheriting ``os.environ`` directly.
"""

import os


def clean_python_env(**overrides: str) -> dict[str, str]:
    """Copy of ``os.environ`` hardened for env-python subprocesses.

    Sets ``PYTHONNOUSERSITE=1`` (keep the user's personal site-packages
    out of ``sys.path``) and drops ``PYTHONPATH`` / ``PYTHONHOME`` (never
    ours; only ever a foreign-interpreter leak). ``overrides`` are applied
    last, so callers can still add e.g. ``PYTHONUNBUFFERED="1"``.
    """
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(overrides)
    return env
