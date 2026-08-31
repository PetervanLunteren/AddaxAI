"""Backend shutdown must take the tracked ML subprocess with it.

`popen_group` puts every analysis subprocess in its own session so that
cancel can kill the whole tree. The flip side is that nothing else kills
it: a detector left behind by an app quit ran to the end on its own and
deleted the checkpoint the next run needed. `kill_all_tracked` is what
the lifespan calls on shutdown.
"""

import subprocess
import sys

from app.core.job_cancellation import (
    clear_cancel,
    kill_all_tracked,
    track_subprocess,
)
from app.core.subprocess_group import popen_group


def _sleeper() -> subprocess.Popen:
    return popen_group(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_kill_all_tracked_kills_a_live_tracked_process():
    proc = _sleeper()
    try:
        with track_subprocess("job-live", proc):
            assert proc.poll() is None
            assert kill_all_tracked() == 1
            assert proc.wait(timeout=10) != 0
    finally:
        clear_cancel("job-live")
        if proc.poll() is None:
            proc.kill()


def test_kill_all_tracked_skips_a_finished_process():
    proc = popen_group([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    try:
        with track_subprocess("job-done", proc):
            assert kill_all_tracked() == 0
    finally:
        clear_cancel("job-done")


def test_kill_all_tracked_with_nothing_tracked_is_a_no_op():
    assert kill_all_tracked() == 0
