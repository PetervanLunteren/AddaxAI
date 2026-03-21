"""Tests for addaxai.ui.advanced.about_tab — write_about_tab."""

import inspect
import pytest


def test_import_write_about_tab():
    from addaxai.ui.advanced.about_tab import write_about_tab


def test_write_about_tab_accepts_about_text_widget():
    from addaxai.ui.advanced.about_tab import write_about_tab
    sig = inspect.signature(write_about_tab)
    assert "about_text_widget" in sig.parameters


def test_write_about_tab_accepts_hyperlink():
    from addaxai.ui.advanced.about_tab import write_about_tab
    sig = inspect.signature(write_about_tab)
    assert "hyperlink" in sig.parameters


def test_write_about_tab_accepts_text_font():
    from addaxai.ui.advanced.about_tab import write_about_tab
    sig = inspect.signature(write_about_tab)
    assert "text_font" in sig.parameters


def test_write_about_tab_accepts_scroll():
    from addaxai.ui.advanced.about_tab import write_about_tab
    sig = inspect.signature(write_about_tab)
    assert "scroll" in sig.parameters


def test_write_about_tab_is_callable():
    from addaxai.ui.advanced.about_tab import write_about_tab
    assert callable(write_about_tab)
