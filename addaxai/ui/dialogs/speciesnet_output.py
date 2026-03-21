"""SpeciesNetOutputWindow — output window for SpeciesNet deployment."""
import logging
import os
import signal
import tkinter as tk
import customtkinter
from subprocess import Popen
from typing import Any, Callable, Optional

from addaxai.utils.files import remove_ansi_escape_sequences

logger = logging.getLogger(__name__)


class SpeciesNetOutputWindow:
    """Toplevel window that shows live SpeciesNet stdout and a Cancel button."""

    def __init__(self, master: Any, bring_to_top_func: Optional[Callable[[Any], None]] = None,
                 on_cancel: Optional[Callable[[], None]] = None) -> None:
        """
        Args:
            master: parent tkinter window
            bring_to_top_func: callable(window) to bring window to front
            on_cancel: callable() invoked when user cancels — responsible for
                       setting cancel flags and re-enabling buttons in the caller
        """
        self.on_cancel = on_cancel
        self.sppnet_output_window_root = customtkinter.CTkToplevel(master)
        self.sppnet_output_window_root.title("SpeciesNet output")
        self.text_area = tk.Text(self.sppnet_output_window_root, wrap=tk.WORD, height=7, width=85)
        self.text_area.pack(padx=10, pady=10)
        self.close_button = tk.Button(self.sppnet_output_window_root, text="Cancel", command=self.cancel)
        self.close_button.pack(pady=5)
        self.sppnet_output_window_root.protocol("WM_DELETE_WINDOW", self.close)
        if bring_to_top_func:
            bring_to_top_func(self.sppnet_output_window_root)

    def add_string(self, text: str, process: Optional[Popen] = None) -> None:
        if process is not None:
            self.process = process
        if text.strip():
            logger.debug(text.rstrip())

            clean_text = remove_ansi_escape_sequences(text)

            # Check if this is a progress bar update
            is_pbar = "%" in clean_text

            if not is_pbar:
                self.text_area.insert(tk.END, clean_text + "\n")
                self.text_area.see(tk.END)
                self.sppnet_output_window_root.update()
                return

            # Ensure attributes exist before updating
            if not hasattr(self, "detector_preprocess_line"):
                self.detector_preprocess_line   = " Detector preprocess:   0%\n"
            if not hasattr(self, "detector_predict_line"):
                self.detector_predict_line      = " Detector predict:      0%\n"
            if not hasattr(self, "classifier_preprocess_line"):
                self.classifier_preprocess_line = " Classifier preprocess: 0%\n"
            if not hasattr(self, "classifier_predict_line"):
                self.classifier_predict_line    = " Classifier predict:    0%\n"
            if not hasattr(self, "geolocation_line"):
                self.geolocation_line           = " Geolocation:           0%\n"

            # Update progress bar lines based on prefixes
            if clean_text.startswith("Detector preprocess"):
                self.detector_preprocess_line = clean_text
            elif clean_text.startswith("Detector predict"):
                self.detector_predict_line = clean_text
            elif clean_text.startswith("Classifier preprocess"):
                self.classifier_preprocess_line = clean_text
            elif clean_text.startswith("Classifier predict"):
                self.classifier_predict_line = clean_text
            elif clean_text.startswith("Geolocation"):
                self.geolocation_line = clean_text

            # Insert all progress bars together to maintain order
            self.text_area.insert(tk.END, f"\n {self.detector_preprocess_line}", "progress")
            self.text_area.insert(tk.END, f" {self.detector_predict_line}", "progress")
            self.text_area.insert(tk.END, f" {self.classifier_preprocess_line}", "progress")
            self.text_area.insert(tk.END, f" {self.classifier_predict_line}", "progress")
            self.text_area.insert(tk.END, f" {self.geolocation_line}", "progress")

            self.text_area.see(tk.END)
            self.sppnet_output_window_root.update()

    def close(self) -> None:
        self.sppnet_output_window_root.destroy()

    def cancel(self) -> None:
        if os.name == 'nt':
            Popen(f"TASKKILL /F /PID {self.process.pid} /T")
        else:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)  # type: ignore[attr-defined]
        if self.on_cancel:
            self.on_cancel()
        self.sppnet_output_window_root.destroy()
