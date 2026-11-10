"""Machine facts for the diagnostics bundle: total RAM and the CPU name.

Both answer questions that came up on a native crash report (2026-10-11,
a detector dying with an access violation on the CPU path): is there
enough memory for the model at all, and is this a CPU without AVX. The
bundle could answer neither. ``platform.processor()`` is not enough: on
Windows it gives the family and model numbers, on macOS the architecture.

Separate from ``process_memory.py`` on purpose. That one asks what this
process uses right now; this one asks what the machine is. Like
``gpu_info.py``, it stays with what the OS ships, so no psutil.

Diagnostics only: never raises, returns None when a value cannot be read.
"""

from __future__ import annotations

import subprocess
import sys

_TIMEOUT_SECONDS = 5


def total_memory_bytes() -> int | None:
    """Physical RAM in bytes, or None if unreadable."""
    try:
        if sys.platform == "win32":
            return _total_memory_windows()
        if sys.platform == "darwin":
            return int(_run(["sysctl", "-n", "hw.memsize"]))
        return _proc_value("/proc/meminfo", "MemTotal", kib=True)
    except Exception:
        return None


def cpu_name() -> str | None:
    """The CPU's marketing name, e.g. "Intel(R) Core(TM) i5-6500", or None."""
    try:
        if sys.platform == "win32":
            return _cpu_name_windows()
        if sys.platform == "darwin":
            return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        return str(_proc_value("/proc/cpuinfo", "model name"))
    except Exception:
        return None


def _run(command: list[str]) -> str:
    """Stripped stdout of `command`; raises when it is missing, fails or hangs."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
        check=True,
    )
    return result.stdout.strip()


def _proc_value(path: str, key: str, *, kib: bool = False) -> int | str:
    """The value of the first `key: value` line in a procfs file.

    With `kib`, the value is a kB figure as /proc/meminfo writes it and
    comes back as an int of bytes.
    """
    with open(path) as f:
        for line in f:
            if line.startswith(key):
                value = line.split(":", 1)[1].strip()
                return int(value.split()[0]) * 1024 if kib else value
    raise KeyError(f"{key} not in {path}")


def _total_memory_windows() -> int:
    import ctypes
    import ctypes.wintypes as wt

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD),
            ("dwMemoryLoad", wt.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = wt.BOOL
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.ullTotalPhys)


def _cpu_name_windows() -> str:
    import winreg

    key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
