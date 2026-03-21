"""PatienceDialog — simple progress window for AddaxAI."""

import math
import tkinter as tk
from tkinter import ttk
from typing import Any

try:
    import customtkinter
    _CTkToplevel = customtkinter.CTkToplevel
except ImportError:
    class _CTkToplevel:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

from addaxai.i18n import t


class PatienceDialog:
    def __init__(self, total: int, text: str, master: Any = None) -> None:
        self.root = _CTkToplevel(master)
        self.root.title(t('be_patient'))
        self.root.geometry("+10+10")
        self.total = total
        self.text = text
        self.label = tk.Label(self.root, text=text)
        self.label.pack(pady=10)
        self.progress = ttk.Progressbar(self.root, mode='determinate', length=200)
        self.progress.pack(pady=10, padx=10)
        self.root.withdraw()

    def open(self) -> None:
        self.root.update()
        self.root.deiconify()

    def update_progress(self, current: int, percentage: bool = False) -> None:
        # updating takes considerable time - only do this 100 times
        if current % math.ceil(self.total / 100) == 0:
            self.progress['value'] = (current / self.total) * 100
            if percentage:
                percentage_value = round((current / self.total) * 100)
                self.label.configure(text=f"{self.text}\n{percentage_value}%")
            else:
                self.label.configure(text=f"{self.text}\n{current} of {self.total}")
            self.root.update()

    def close(self) -> None:
        self.root.destroy()
