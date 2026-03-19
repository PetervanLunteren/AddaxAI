"""Smoke tests to verify the addaxai package structure imports cleanly.

These tests run without any heavy dependencies (no tkinter, no ML frameworks).
They exist to catch broken imports during refactoring.
"""

import importlib
import pytest


PACKAGES = [
    "addaxai",
    "addaxai.core",
    "addaxai.models",
    "addaxai.processing",
    "addaxai.analysis",
    "addaxai.i18n",
    "addaxai.ui",
    "addaxai.ui.advanced",
    "addaxai.ui.simple",
    "addaxai.ui.dialogs",
    "addaxai.ui.widgets",
    "addaxai.hitl",
    "addaxai.utils",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports(package):
    """Every subpackage should be importable without errors."""
    mod = importlib.import_module(package)
    assert mod is not None
