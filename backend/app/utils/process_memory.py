"""Current process memory (RSS) for diagnostic log lines.

Exists because the save-outputs worker once died on a user's machine
with no trace of which module was running or how memory developed
(Wayne's 2026-08 report). psutil would be a new dependency plus a
backend.spec entry, so this stays with what the OS ships: `ps` on
macOS / Linux, Win32 GetProcessMemoryInfo via ctypes on Windows.

Diagnostics only: never raises, returns None when the value cannot be
read, and must never gate program behavior.
"""

from __future__ import annotations

import os
import subprocess
import sys


def rss_mb() -> float | None:
    """Resident set size of this process in MB, or None if unreadable."""
    try:
        if sys.platform == "win32":
            return _rss_mb_windows()
        return _rss_mb_ps()
    except Exception:
        return None


def _rss_mb_ps() -> float | None:
    """`ps -o rss=` reports RSS in KB on both macOS and Linux."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    if not out:
        return None
    return int(out) / 1024


def _rss_mb_windows() -> float | None:
    import ctypes
    import ctypes.wintypes as wt

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD),
            ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # argtypes are load-bearing: without them ctypes passes the
    # GetCurrentProcess() pseudo-handle (-1) as a 32-bit int, which
    # truncates the 64-bit HANDLE and fails with ERROR_INVALID_HANDLE.
    # Verified on a real Windows box (2026-08-12): untyped call
    # returned error 6, this form returns the working set.
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wt.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wt.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ok = psapi.GetProcessMemoryInfo(
        wt.HANDLE(-1),  # GetCurrentProcess() pseudo-handle
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return None
    return counters.WorkingSetSize / (1024 * 1024)
