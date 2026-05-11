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

import sys
from pathlib import Path

# FILE_ATTRIBUTE_HIDDEN from windows.h.
_FILE_ATTRIBUTE_HIDDEN = 0x02


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
    """
    path.mkdir(parents=parents, exist_ok=exist_ok)
    if sys.platform != "win32":
        return path
    for candidate in [path, *path.parents]:
        if candidate.name == ".addaxai":
            set_windows_hidden(candidate)
            break
    return path
