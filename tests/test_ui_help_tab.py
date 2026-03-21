"""Tests for addaxai.ui.advanced.help_tab — HyperlinkManager and write_help_tab."""

import inspect
import pytest


def test_import_hyperlink_manager():
    from addaxai.ui.advanced.help_tab import HyperlinkManager


def test_import_write_help_tab():
    from addaxai.ui.advanced.help_tab import write_help_tab


def test_hyperlink_manager_accepts_text():
    from addaxai.ui.advanced.help_tab import HyperlinkManager
    sig = inspect.signature(HyperlinkManager.__init__)
    assert "text" in sig.parameters


def test_hyperlink_manager_accepts_green_primary():
    from addaxai.ui.advanced.help_tab import HyperlinkManager
    sig = inspect.signature(HyperlinkManager.__init__)
    assert "green_primary" in sig.parameters


def test_hyperlink_manager_has_add_method():
    from addaxai.ui.advanced.help_tab import HyperlinkManager
    assert hasattr(HyperlinkManager, "add")
    assert hasattr(HyperlinkManager, "reset")


def test_write_help_tab_accepts_help_text_widget():
    from addaxai.ui.advanced.help_tab import write_help_tab
    sig = inspect.signature(write_help_tab)
    assert "help_text_widget" in sig.parameters


def test_write_help_tab_accepts_hyperlink():
    from addaxai.ui.advanced.help_tab import write_help_tab
    sig = inspect.signature(write_help_tab)
    assert "hyperlink" in sig.parameters


def test_write_help_tab_accepts_text_font():
    from addaxai.ui.advanced.help_tab import write_help_tab
    sig = inspect.signature(write_help_tab)
    assert "text_font" in sig.parameters
