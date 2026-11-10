"""Clean environment for spawned Python subprocesses.

AddaxAI's ML envs are micromamba envs, not venvs, so their pythons keep
user-site enabled: anything the user ever ``pip install --user``-ed for a
matching Python minor version lands on ``sys.path`` BEFORE the env's own
site-packages and shadows the pinned packages (the classic "conda env
broken by ~/.local" failure). A globally exported ``PYTHONPATH`` (common
on Anaconda setups) leaks the same way with higher precedence.

Every spawn of an env python or micromamba must therefore go through
``clean_python_env()`` instead of inheriting ``os.environ`` directly,
and read its output with ``encoding="utf-8", errors="replace"`` (see
``PYTHONIOENCODING`` below).
"""

import os


def clean_python_env(**overrides: str) -> dict[str, str]:
    """Copy of ``os.environ`` hardened for env-python subprocesses.

    Sets ``PYTHONNOUSERSITE=1`` (keep the user's personal site-packages
    out of ``sys.path``) and drops ``PYTHONPATH`` / ``PYTHONHOME`` (never
    ours; only ever a foreign-interpreter leak). ``overrides`` are applied
    last, so callers can still add e.g. ``CUDA_VISIBLE_DEVICES``.

    ``PYTHONFAULTHANDLER=1`` makes a native crash (access violation,
    segfault) print the Python stack of every thread to stderr before the
    process dies, so backend.log shows which call was running instead of
    only the exit code. ``PYTHONUNBUFFERED=1`` keeps the child's prints
    out of a block buffer that a crash would discard. Both are read by
    python only; micromamba runs through here too and ignores them, but
    the pip it runs inherits them.

    ``PYTHONIOENCODING=utf-8`` is the child's half of a pipe contract:
    every reader of these pipes passes ``encoding="utf-8",
    errors="replace"``. Left to the locale, both ends guess the Windows
    code page (cp932, cp1252), which cannot hold tqdm's bar characters or
    a user path in another script, and a child that inherited the
    variable as utf-8 from the user's shell failed the parent's decode
    on its first non-ASCII byte. Set unconditionally, over any inherited
    value. ``PYTHONIOENCODING``, not ``PYTHONUTF8``: the latter is
    ignored for stdio whenever the former is set (PEP 540).
    """
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(overrides)
    return env
