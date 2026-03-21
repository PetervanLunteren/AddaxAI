"""Tests for addaxai.ui.dialogs.progress — ProgressWindow class."""

import inspect
import pytest


def test_import_progress_window():
    from addaxai.ui.dialogs.progress import ProgressWindow


def test_progresswindow_accepts_master():
    from addaxai.ui.dialogs.progress import ProgressWindow
    sig = inspect.signature(ProgressWindow.__init__)
    assert "master" in sig.parameters


def test_progresswindow_accepts_scale_factor():
    from addaxai.ui.dialogs.progress import ProgressWindow
    sig = inspect.signature(ProgressWindow.__init__)
    assert "scale_factor" in sig.parameters


def test_progresswindow_accepts_padx_pady():
    from addaxai.ui.dialogs.progress import ProgressWindow
    sig = inspect.signature(ProgressWindow.__init__)
    assert "padx" in sig.parameters
    assert "pady" in sig.parameters


def test_progresswindow_accepts_green_primary():
    from addaxai.ui.dialogs.progress import ProgressWindow
    sig = inspect.signature(ProgressWindow.__init__)
    assert "green_primary" in sig.parameters


def test_progresswindow_has_required_methods():
    from addaxai.ui.dialogs.progress import ProgressWindow
    assert hasattr(ProgressWindow, "update_values")
    assert hasattr(ProgressWindow, "open")
    assert hasattr(ProgressWindow, "close")
