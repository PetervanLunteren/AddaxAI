"""
Best-effort recursive deletion.

Lives here rather than in a router so services can reuse it without
importing from the API layer. `EnvironmentManager._safe_rmtree` is a
deliberately different thing: it re-raises so a failed env wipe aborts
the install. This one swallows, because its callers are wipes where a
single locked file must not stop the rest.
"""

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def safe_rmtree(path: Path) -> bool:
    """
    Best-effort recursive removal.

    Tolerates per-file failures so a single locked or read-only file
    does not abort the whole wipe and leave envs / models in a
    half-deleted state. Real-world cases this protects against:

    - Windows: the running backend's `logs/backend.log` is held open by
      its own logging handler. Without the swallow, `shutil.rmtree`
      raised on it and the surrounding for loop did continue to the
      next dir, BUT any file *inside* the failed dir that was already
      partway through deletion left the dir in a broken state. This is
      what corrupted env-pytorch in a beta tester's diag bundle: Lib/
      was deleted but python.exe survived, so `_validate_env` later
      reported the env as healthy and the classification worker
      crashed with `ModuleNotFoundError: encodings`.

    - Read-only attribute on Windows files (set by some installers,
      antivirus quarantine restores, copy-from-network-share). chmod +
      retry handles those instead of letting them block the wipe.

    Symlinks (and Windows directory junctions, which `is_symlink()`
    also reports) are unlinked, never followed, so removing a link
    never touches the tree it points at.

    Returns True only when the path is fully gone after the call.
    Caller can therefore trust the returned list as "actually removed",
    not "removal attempted".
    """
    if not path.exists() and not path.is_symlink():
        return False

    def _onerror(func: Callable, p: str, _exc_info: tuple) -> None:
        # Try clearing the read-only attribute and retrying. Most
        # Windows EACCES failures are this. If it still fails, swallow
        # so rmtree keeps walking the rest of the tree.
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
            return
        except OSError as e:
            logger.warning(f"Could not remove {p}: {e}")

    if path.is_symlink():
        # A symlink to a directory needs rmdir on Windows and unlink
        # elsewhere. Try both rather than branching on platform.
        for remove in (os.unlink, os.rmdir):
            try:
                remove(path)
                return True
            except OSError:
                continue
        logger.warning(f"Could not remove link {path}")
        return False

    if path.is_file():
        try:
            try:
                path.chmod(stat.S_IWRITE)
            except OSError:
                pass
            path.unlink()
        except OSError as e:
            logger.warning(f"Could not remove {path}: {e}")
            return False
        return True

    shutil.rmtree(path, onerror=_onerror)
    return not path.exists()
