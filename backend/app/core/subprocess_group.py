"""Popen wrapper that puts children in their own process group / job object.

Needed so we can kill an ML subprocess *and everything it spawns* (shell
grandchildren, torch workers, ffmpeg, ...) in one move when a user
cancels a running job.

Usage:
    from app.core.subprocess_group import popen_group
    proc = popen_group([sys.executable, "worker.py"], stdout=subprocess.PIPE)
"""

import os
import subprocess


def popen_group(*args, **kwargs) -> subprocess.Popen:
    """subprocess.Popen that creates a new session (Unix) or process
    group (Windows) so the whole descendant tree can be killed later.

    Caller still owns the Popen; this helper only sets the flags that
    are needed for a later `killpg` / `taskkill /T` to reach children.
    """
    if os.name == "nt":
        flags = kwargs.get("creationflags", 0)
        kwargs["creationflags"] = flags | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(*args, **kwargs)
