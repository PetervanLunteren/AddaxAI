"""Tests for addaxai.ui.widgets — widget classes."""

import inspect
import pytest

try:
    import customtkinter
    _HAS_CTK = True
except ImportError:
    _HAS_CTK = False


# --- Import tests (always run) ---

def test_import_frames():
    from addaxai.ui.widgets.frames import MyMainFrame, MySubFrame, MySubSubFrame


def test_import_buttons():
    from addaxai.ui.widgets.buttons import InfoButton, CancelButton, GreyTopButton


def test_import_species_selection():
    from addaxai.ui.widgets.species_selection import SpeciesSelectionFrame


# --- Signature tests (always run, no tkinter needed) ---

def test_mymainframe_accepts_scale_factor():
    from addaxai.ui.widgets.frames import MyMainFrame
    sig = inspect.signature(MyMainFrame.__init__)
    assert "scale_factor" in sig.parameters


def test_greytopbutton_accepts_color_keywords():
    from addaxai.ui.widgets.buttons import GreyTopButton
    sig = inspect.signature(GreyTopButton.__init__)
    assert "yellow_secondary" in sig.parameters
    assert "yellow_tertiary" in sig.parameters
    assert "border_width" in sig.parameters


def test_speciesselectionframe_accepts_dummy_spp():
    from addaxai.ui.widgets.species_selection import SpeciesSelectionFrame
    sig = inspect.signature(SpeciesSelectionFrame.__init__)
    assert "dummy_spp" in sig.parameters


def test_speciesselectionframe_accepts_pady():
    from addaxai.ui.widgets.species_selection import SpeciesSelectionFrame
    sig = inspect.signature(SpeciesSelectionFrame.__init__)
    assert "pady" in sig.parameters


def test_mymainframe_scale_factor_default_is_one():
    from addaxai.ui.widgets.frames import MyMainFrame
    sig = inspect.signature(MyMainFrame.__init__)
    assert sig.parameters["scale_factor"].default == 1.0


def test_greytopbutton_color_defaults():
    from addaxai.ui.widgets.buttons import GreyTopButton
    sig = inspect.signature(GreyTopButton.__init__)
    assert sig.parameters["yellow_secondary"].default == "#F0EEDC"
    assert sig.parameters["yellow_tertiary"].default == "#E4E1D0"
    assert sig.parameters["border_width"].default == 0
