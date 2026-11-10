"""
Windows Smart App Control detection.

Smart App Control (SAC) is a Windows 11 code-integrity feature that only
loads binaries that are validly signed or that Microsoft's cloud service
rates reputable. The analysis environments are built on the user's own
machine from conda-forge and PyPI, and most of their compiled files are
unsigned, so SAC can block them file by file. Which files is decided by
the reputation service, not by us: on the machine that surfaced this
(2026-09-25) the stdlib .pyds loaded fine while matplotlib's and
kiwisolver's were blocked, so the env boot probe passed and the run died
with a bare `ModuleNotFoundError` deep inside a worker subprocess.

The app cannot fix the blocking itself: the files only exist after setup
and signing them would need a private key we cannot ship. The one useful
thing is to *name* SAC when an unexplained analysis failure happens on a
machine where it is enforced, so the user reads "Smart App Control" in
the error instead of digging through Event Viewer. The user-facing fix
lives in docs/help/locked-down-computers.mdx.

The state is one registry value, readable without admin rights. These
helpers are deliberately best-effort: they ride on failure paths and a
diagnostics collector, so a registry surprise must degrade to "unknown",
never raise.
"""

import sys

from app.core.logging_config import get_logger

logger = get_logger(__name__)

_POLICY_KEY = r"SYSTEM\CurrentControlSet\Control\CI\Policy"
_POLICY_VALUE = "VerifiedAndReputablePolicyState"
# 2 ("evaluation") observes without ever blocking, so only 1 earns a hint.
_STATES = {0: "off", 1: "on", 2: "evaluation"}

SMART_APP_CONTROL_HINT = (
    " This computer has Windows Smart App Control turned on, which can "
    "block parts of the analysis software. How to check and fix it: "
    "https://docs.addaxai.com/docs/help/locked-down-computers"
    "#windows-smart-app-control-blocks-the-analysis"
)


def _read_policy_value() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _POLICY_KEY) as key:
            value, _ = winreg.QueryValueEx(key, _POLICY_VALUE)
        return int(value)
    except OSError:
        # Key or value absent: Windows 10, or server SKUs without SAC.
        return None
    except (TypeError, ValueError):
        return None


def smart_app_control_state() -> str | None:
    """"off", "on" or "evaluation"; None off Windows or when unreadable."""
    value = _read_policy_value()
    if value is None:
        return None
    state = _STATES.get(value)
    if state is None:
        logger.warning(f"Unknown Smart App Control policy value: {value}")
    return state


def with_smart_app_control_hint(error: str) -> str:
    """Append the SAC hint to an unexplained failure message.

    Only when SAC is enforced; everywhere else the message comes back
    unchanged. For failures whose cause is already known and stated
    (missing folder, server restart), keep the clean message and do not
    call this.
    """
    if smart_app_control_state() != "on":
        return error
    error = error.rstrip()
    separator = "" if error.endswith((".", "!", "?")) else "."
    return f"{error}{separator}{SMART_APP_CONTROL_HINT}"
