"""
Find and remove a legacy AddaxAI install (v5 / v6).

Legacy AddaxAI (https://github.com/PetervanLunteren/AddaxAI, v6.37 at the
time of writing) installs to completely different locations than this
app, so upgrading leaves two full installs alive on the machine, the old
one holding 10 to 30 GB of conda envs and model weights that will never
be used again. This module finds the old one and deletes it.

Why this is not in the installers: macOS ships a dmg and drag-to-
Applications runs no code, and the Linux deb postinst runs as root so it
cannot know which user's home to clean (`$SUDO_USER` is unset under
App Center / PackageKit). Only the Windows NSIS installer could do it.
One Python implementation running as the logged-in user covers all
three platforms, which is also the right permission context.

Legacy install roots:

    Windows   %USERPROFILE%\\AddaxAI_files
    macOS     /Applications/AddaxAI_files
    Linux     ~/.AddaxAI_files

Legacy writes nothing outside that tree apart from a desktop shortcut
(and an icon on Linux). Its analysis outputs live in the user's own
image folders and a destination folder they picked, so nothing here
touches user data.
"""

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.logging_config import get_logger
from app.utils.fs_remove import safe_rmtree

logger = get_logger(__name__)

# Legacy's GUI entry point. Its presence is the single rule for "a
# legacy install lives here", on every platform. Keying off the folder
# name alone would be wrong on Windows, where our own installer creates
# `AddaxAI_files` just to hold the Timelapse shim.
_MARKER = Path("AddaxAI") / "AddaxAI_GUI.py"

# Legacy's own version file, the same one its macOS installer reads to
# report which version it is replacing.
_VERSION_FILE = Path("AddaxAI") / "version.txt"

# Windows only. Our NSIS installer writes a Timelapse launcher shim to
# `%USERPROFILE%\AddaxAI_files\AddaxAI\open.bat`, which is inside the
# legacy install root. Timelapse still looks for that path, so this one
# file has to survive the purge. See electron/build/installer.nsh.
_SHIM = Path("AddaxAI") / "open.bat"


@dataclass(frozen=True)
class LegacyScan:
    """What a scan found. All paths are absolute."""

    root: Path | None = None
    junction: Path | None = None
    version: str | None = None
    # Legacy installs we can see but cannot delete without elevation.
    manual: tuple[Path, ...] = ()

    @property
    def removable(self) -> list[Path]:
        return [p for p in (self.root, self.junction) if p is not None]

    @property
    def found(self) -> bool:
        return bool(self.removable) or bool(self.manual)


def scan() -> LegacyScan:
    """
    Look for a legacy install. Cheap: a couple of `is_file()` calls, so
    it is fine to run on every app launch.

    Deliberately does not stat the desktop shortcut. On macOS, reading
    `~/Desktop` triggers a permission prompt for a non-sandboxed app,
    and firing that on every launch for every user (including everyone
    who never had legacy installed) is not acceptable. The shortcut is
    handled in `remove()`, where the prompt reads as a consequence of
    what the user just asked for.
    """
    root = _user_root()
    if not _is_legacy_install(root):
        root = None

    manual = _manual_root()
    if manual is not None and not _is_legacy_install(manual):
        manual = None

    return LegacyScan(
        root=root,
        # Only offered alongside a real install, so an orphaned link can
        # never make the app prompt about a legacy version that is gone.
        junction=_junction() if root is not None else None,
        version=_read_version(root) if root is not None else None,
        manual=(manual,) if manual is not None else (),
    )


def remove() -> list[Path]:
    """
    Delete the legacy install.

    Returns the paths that survived. An empty list means the purge
    worked. Anything else means files were locked (legacy still
    running, antivirus, an open Explorer window) and the caller should
    tell the user to close the old app and retry. We check the result
    rather than trying to detect a running process first: one rule that
    covers every cause, on every platform, with no extra dependency.
    """
    found = scan()
    survivors: list[Path] = []

    if found.root is not None:
        _purge_root(found.root)
        # Same marker as detection, so "did it work" and "is it there"
        # can never disagree. On Windows the root itself survives on
        # purpose, holding nothing but the Timelapse shim.
        if _is_legacy_install(found.root):
            survivors.append(found.root)

    if found.junction is not None and _present(found.junction):
        _remove(found.junction)
        if _present(found.junction):
            survivors.append(found.junction)

    # Desktop leftovers last, and only here. See the note in scan().
    for leftover in _desktop_leftovers():
        if _present(leftover):
            safe_rmtree(leftover)

    logger.warning(
        f"Legacy AddaxAI removal: root={found.root} junction={found.junction} "
        f"survivors={survivors}"
    )
    return survivors


# ---------------------------------------------------------------------
# Platform paths
# ---------------------------------------------------------------------


def _user_root() -> Path:
    """The legacy install root for this platform."""
    if sys.platform == "win32":
        return Path.home() / "AddaxAI_files"
    if sys.platform == "darwin":
        return Path("/Applications/AddaxAI_files")
    return Path.home() / ".AddaxAI_files"


def _junction() -> Path | None:
    """
    Windows only. The legacy installer makes `EcoAssist_files` a junction
    pointing at `AddaxAI_files`, so old EcoAssist shortcuts keep working.
    """
    if sys.platform != "win32":
        return None
    return Path.home() / "EcoAssist_files"


def _manual_root() -> Path | None:
    """
    Windows only. Legacy's own docs describe moving the install to
    Program Files by hand for multi-user machines. We run unelevated so
    we can only report it.
    """
    if sys.platform != "win32":
        return None
    program_files = os.environ.get("ProgramFiles")
    return Path(program_files) / "AddaxAI_files" if program_files else None


def _desktop_leftovers() -> list[Path]:
    """
    Shortcuts and icons the legacy installer dropped outside its own
    tree. Windows is absent on purpose: legacy's desktop shortcut is
    also called `AddaxAI.lnk`, so our own installer already replaced it,
    and deleting it would delete ours.
    """
    home = Path.home()
    if sys.platform == "darwin":
        return [home / "Desktop" / "AddaxAI.app"]
    if sys.platform == "win32":
        return []
    return [
        home / "Desktop" / "Linux_open_AddaxAI_shortcut.desktop",
        home / ".icons" / "logo_small_bg.png",
    ]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _is_legacy_install(root: Path | None) -> bool:
    return root is not None and (root / _MARKER).is_file()


def _read_version(root: Path) -> str | None:
    try:
        return (root / _VERSION_FILE).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _present(path: Path) -> bool:
    """True if the path exists, including a link whose target is gone."""
    return path.exists() or path.is_symlink()


def _purge_root(root: Path) -> None:
    """Delete the legacy install tree, keeping the Timelapse shim."""
    if sys.platform != "win32":
        _remove(root)
        return

    # Windows: everything goes except `AddaxAI\open.bat`, which is ours.
    keep = root / _SHIM
    for child in _children(root):
        if child != keep.parent:
            _remove(child)
            continue
        for grandchild in _children(child):
            if grandchild != keep:
                _remove(grandchild)


def _children(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError as e:
        logger.warning(f"Could not list {path}: {e}")
        return []


def _remove(path: Path) -> None:
    """Delete a file, directory or link. Never follows a junction."""
    if _is_junction(path):
        # Dropping the reparse point leaves the directory it points at
        # untouched. `safe_rmtree` would too, but `shutil.rmtree` raises
        # on a link, so handle it explicitly.
        try:
            os.rmdir(path)
        except OSError as e:
            logger.warning(f"Could not remove junction {path}: {e}")
        return
    safe_rmtree(path)


def _is_junction(path: Path) -> bool:
    """
    True for a Windows directory junction.

    `os.path.isjunction()` does this in one call but landed in Python
    3.12, and the frozen build runs 3.11 (.github/workflows/build-electron.yml).
    """
    if os.name != "nt":
        return False
    try:
        return os.lstat(path).st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
    except (OSError, AttributeError):
        return False
