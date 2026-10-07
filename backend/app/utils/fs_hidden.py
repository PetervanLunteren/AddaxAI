"""
Cross-platform hidden-folder helpers for the `.addaxai` artifact root.

macOS and Linux file managers respect the leading-dot convention and
hide `.addaxai` automatically. Windows ignores leading dots and uses a
separate HIDDEN file attribute, so we set it explicitly here to keep
the artifact folder out of Explorer's default view. The user can still
surface it via "Show hidden items" when they need to inspect it.

Setting the attribute is cosmetic: failures are swallowed because the
pipeline keeps working regardless of folder visibility.
"""

from __future__ import annotations

import errno
import sys
from pathlib import Path

# FILE_ATTRIBUTE_HIDDEN from windows.h.
_FILE_ATTRIBUTE_HIDDEN = 0x02

# A read-only drive (EROFS: on a Mac, any NTFS drive), a folder the user
# may not write to (EACCES), or macOS refusing the app access to the
# volume (EPERM). The raw "[Errno 30] Read-only file system" told a user
# nothing about the cause or the fix (2026-10-05).
_NOT_WRITABLE_ERRNOS = frozenset({errno.EROFS, errno.EACCES, errno.EPERM})

FOLDER_NOT_WRITABLE_MESSAGE = (
    "AddaxAI cannot write to this folder. It saves a hidden .addaxai folder "
    "next to your files and never changes the files themselves. A Mac "
    "cannot write to drives formatted for Windows (NTFS). How to fix it: "
    "https://docs.addaxai.com/docs/help/faq"
    "#the-analysis-fails-because-addaxai-cannot-write-to-the-folder"
)


def set_windows_hidden(path: Path) -> None:
    """Mark `path` as hidden via Win32 SetFileAttributesW.

    No-op on non-Windows and on any failure. The attribute is cosmetic
    and the pipeline does not depend on it.
    """
    if sys.platform != "win32" or not path.exists():
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetFileAttributesW(
            str(path), _FILE_ATTRIBUTE_HIDDEN
        )
    except Exception:
        pass


def mkdir_hidden_addaxai(
    path: Path, *, parents: bool = True, exist_ok: bool = True
) -> Path:
    """`path.mkdir(...)` plus, on Windows, set HIDDEN on the `.addaxai`
    segment within the new path.

    Hiding the artifact root is enough: subfolders inside it inherit
    visibility (the user has to enter `.addaxai` to see them). Setting
    HIDDEN on a folder that already has it is harmless.

    A folder AddaxAI may not write to raises OSError with
    `FOLDER_NOT_WRITABLE_MESSAGE`, chained to the original error.
    """
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except OSError as e:
        if e.errno not in _NOT_WRITABLE_ERRNOS:
            raise
        # One argument, so str() is the sentence and not "[Errno 30] ...";
        # the original stays in brackets (path) and as __cause__ (errno).
        raise OSError(f"{FOLDER_NOT_WRITABLE_MESSAGE} ({e})") from e
    if sys.platform != "win32":
        return path
    for candidate in [path, *path.parents]:
        if candidate.name == ".addaxai":
            set_windows_hidden(candidate)
            break
    return path
