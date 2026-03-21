"""Tests for addaxai.ui.dialogs — dialog classes."""

import inspect
import pytest


# --- Import tests ---

def test_import_text_button():
    from addaxai.ui.dialogs.text_button import TextButtonWindow


def test_import_patience():
    from addaxai.ui.dialogs.patience import PatienceDialog


def test_import_custom_window():
    from addaxai.ui.dialogs.custom_window import CustomWindow


def test_import_download_progress():
    from addaxai.ui.dialogs.download_progress import (
        EnvDownloadProgressWindow,
        ModelDownloadProgressWindow,
    )


def test_import_info_frames():
    from addaxai.ui.dialogs.info_frames import ModelInfoFrame, DonationPopupFrame


# --- Signature tests ---

def test_textbuttonwindow_accepts_master():
    from addaxai.ui.dialogs.text_button import TextButtonWindow
    sig = inspect.signature(TextButtonWindow.__init__)
    assert "master" in sig.parameters


def test_textbuttonwindow_accepts_bring_to_top_func():
    from addaxai.ui.dialogs.text_button import TextButtonWindow
    sig = inspect.signature(TextButtonWindow.__init__)
    assert "bring_to_top_func" in sig.parameters


def test_patiencedialog_accepts_master():
    from addaxai.ui.dialogs.patience import PatienceDialog
    sig = inspect.signature(PatienceDialog.__init__)
    assert "master" in sig.parameters


def test_customwindow_accepts_master():
    from addaxai.ui.dialogs.custom_window import CustomWindow
    sig = inspect.signature(CustomWindow.__init__)
    assert "master" in sig.parameters


def test_envdownloadprogresswindow_accepts_all_params():
    from addaxai.ui.dialogs.download_progress import EnvDownloadProgressWindow
    sig = inspect.signature(EnvDownloadProgressWindow.__init__)
    for param in ("master", "scale_factor", "padx", "pady", "green_primary", "open_nosleep_func"):
        assert param in sig.parameters, f"Missing param: {param}"


def test_modeldownloadprogresswindow_accepts_all_params():
    from addaxai.ui.dialogs.download_progress import ModelDownloadProgressWindow
    sig = inspect.signature(ModelDownloadProgressWindow.__init__)
    for param in ("master", "scale_factor", "padx", "pady", "green_primary", "open_nosleep_func"):
        assert param in sig.parameters, f"Missing param: {param}"


def test_modelinfoframe_accepts_scale_factor():
    from addaxai.ui.dialogs.info_frames import ModelInfoFrame
    sig = inspect.signature(ModelInfoFrame.__init__)
    assert "scale_factor" in sig.parameters


def test_donationpopupframe_accepts_scale_factor():
    from addaxai.ui.dialogs.info_frames import DonationPopupFrame
    sig = inspect.signature(DonationPopupFrame.__init__)
    assert "scale_factor" in sig.parameters
