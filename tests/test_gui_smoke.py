r"""
GUI smoke test: launches AddaxAI_GUI.py via dev_launch.py, waits for startup,
asserts it hasn't crashed, then terminates it.

Requires the installed env-base Python -- skipped automatically if not found.
Run with:
    C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_smoke.py -v
Or from .venv (test will be skipped if env-base not found):
    .venv/Scripts/python -m pytest tests/test_gui_smoke.py -v
"""

import os
import subprocess
import sys
import time

import pytest

ENV_BASE_PYTHON = r"C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe"
LAUNCHER = os.path.join(os.path.dirname(__file__), "..", "dev_launch.py")
STARTUP_WAIT_SECONDS = 10  # time to allow the GUI to initialize


@pytest.fixture
def env_base_python():
    if not os.path.isfile(ENV_BASE_PYTHON):
        pytest.skip(f"env-base Python not found at {ENV_BASE_PYTHON}")
    return ENV_BASE_PYTHON


def test_gui_starts_without_crashing(env_base_python):
    """GUI process must still be alive after STARTUP_WAIT_SECONDS seconds."""
    proc = subprocess.Popen(
        [env_base_python, os.path.abspath(LAUNCHER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(STARTUP_WAIT_SECONDS)
    exit_code = proc.poll()

    proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    if exit_code is not None:
        decoded_stderr = stderr.decode(errors="replace")
        pytest.fail(
            f"GUI exited early with code {exit_code}.\n"
            f"--- stderr ---\n{decoded_stderr}"
        )
