"""
Smart App Control detection and the failure-message hint.

The registry read itself cannot run on CI (no winreg off Windows), so
these tests pin the mapping and the hint logic around a monkeypatched
`_read_policy_value`, plus the real no-op path on non-Windows. The actual
registry read is verified by hand on a Windows machine:

    python -c "import winreg; k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
    r'SYSTEM\\CurrentControlSet\\Control\\CI\\Policy');
    print(winreg.QueryValueEx(k, 'VerifiedAndReputablePolicyState'))"
"""

import sys

import pytest

from app.utils import windows_security as ws


@pytest.mark.parametrize(
    ("value", "state"),
    [
        (0, "off"),
        (1, "on"),
        (2, "evaluation"),
        (None, None),
        (3, None),  # a value nothing documented predicts reads as unknown
    ],
)
def test_policy_value_maps_to_state(monkeypatch, value, state):
    monkeypatch.setattr(ws, "_read_policy_value", lambda: value)
    assert ws.smart_app_control_state() == state


@pytest.mark.skipif(sys.platform == "win32", reason="real registry on Windows")
def test_off_windows_the_state_is_unknown_without_mocking():
    assert ws.smart_app_control_state() is None


@pytest.mark.parametrize("value", [0, 2, None])
def test_no_hint_unless_enforced(monkeypatch, value):
    """Evaluation mode never blocks anything, so it earns no hint."""
    monkeypatch.setattr(ws, "_read_policy_value", lambda: value)
    assert ws.with_smart_app_control_hint("boom") == "boom"


def test_the_hint_is_appended_as_its_own_sentence(monkeypatch):
    monkeypatch.setattr(ws, "_read_policy_value", lambda: 1)
    hinted = ws.with_smart_app_control_hint("Video detection failed with exit code 1")
    assert hinted.startswith("Video detection failed with exit code 1. This computer")
    assert "Smart App Control" in hinted
    # The link is the part a user can act on, and it must open the right
    # question on the docs page, not just the page.
    assert (
        "https://docs.addaxai.com/docs/help/locked-down-computers"
        "#windows-smart-app-control-blocks-the-analysis" in hinted
    )


def test_an_error_already_ending_a_sentence_gets_no_double_period(monkeypatch):
    monkeypatch.setattr(ws, "_read_policy_value", lambda: 1)
    hinted = ws.with_smart_app_control_hint("Output JSON was not created.")
    assert ".." not in hinted
    assert hinted.startswith("Output JSON was not created. This computer")
