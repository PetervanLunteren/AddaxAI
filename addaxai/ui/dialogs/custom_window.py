"""CustomWindow — generic pop-up window for AddaxAI."""

import tkinter as tk
from typing import Any, Optional

try:
    import customtkinter
    _CTkToplevel = customtkinter.CTkToplevel
except ImportError:
    class _CTkToplevel:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


class CustomWindow:
    def __init__(self, title: str = "", text: str = "", master: Any = None) -> None:
        self.title = title
        self.text = text
        self._master = master
        self.root: Optional[Any] = None

    def open(self) -> None:
        self.root = _CTkToplevel(self._master)
        self.root.title(self.title)
        self.root.geometry("+10+10")

        label = tk.Label(self.root, text=self.text)
        label.pack(padx=10, pady=10)

        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        self.root.destroy()
