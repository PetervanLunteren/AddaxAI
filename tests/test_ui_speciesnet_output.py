"""Tests for addaxai.ui.dialogs.speciesnet_output (Phase 4.7)."""
import inspect
import pytest

try:
    import customtkinter
    from addaxai.ui.dialogs.speciesnet_output import SpeciesNetOutputWindow
    _HAS_CTK = True
except ImportError:
    _HAS_CTK = False
    SpeciesNetOutputWindow = None


pytestmark = pytest.mark.skipif(not _HAS_CTK, reason="customtkinter not installed")


def test_importable():
    assert SpeciesNetOutputWindow is not None


def test_has_required_methods():
    assert hasattr(SpeciesNetOutputWindow, "add_string")
    assert hasattr(SpeciesNetOutputWindow, "close")
    assert hasattr(SpeciesNetOutputWindow, "cancel")


def test_init_accepts_keyword_args():
    sig = inspect.signature(SpeciesNetOutputWindow.__init__)
    params = list(sig.parameters.keys())
    assert "master" in params
    assert "bring_to_top_func" in params
    assert "on_cancel" in params


@pytest.fixture
def root():
    import tkinter as tk
    try:
        r = tk.Tk()
        r.withdraw()
    except Exception:
        pytest.skip("tkinter display not available")
    yield r
    r.destroy()


def test_instantiation_with_master(root):
    win = SpeciesNetOutputWindow(master=root)
    assert win is not None
    win.close()


def test_on_cancel_callback(root):
    called = []
    def _cb():
        called.append(True)
    win = SpeciesNetOutputWindow(master=root, on_cancel=_cb)
    if win.on_cancel:
        win.on_cancel()
    assert called == [True]
    win.close()
