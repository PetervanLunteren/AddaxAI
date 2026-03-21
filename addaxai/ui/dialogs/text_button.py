"""TextButtonWindow dialog for AddaxAI."""

import tkinter as tk
from typing import Any, Callable, List, Optional

try:
    import customtkinter
    _CTkToplevel = customtkinter.CTkToplevel
except ImportError:
    class _CTkToplevel:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


class TextButtonWindow:
    def __init__(self, title: str, text: str, buttons: List[str],
                 master: Any = None, bring_to_top_func: Optional[Callable[[Any], None]] = None) -> None:
        self.root = _CTkToplevel(master)
        self.root.title(title)
        self.root.geometry("+10+10")
        if bring_to_top_func:
            bring_to_top_func(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.user_close)

        self.text_label = tk.Label(self.root, text=text)
        self.text_label.pack(padx=10, pady=10)

        self.selected_button = None
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(padx=10, pady=10)

        for button_text in buttons:
            button = tk.Button(self.button_frame, text=button_text,
                               command=lambda btn=button_text: self._button_click(btn))
            button.pack(side=tk.LEFT, padx=5)

    def _button_click(self, button_text: str) -> None:
        self.selected_button = button_text
        self.root.quit()

    def open(self) -> None:
        self.root.mainloop()

    def user_close(self) -> None:
        self.selected_button = "EXIT"
        self.root.quit()
        self.root.destroy()

    def run(self) -> Optional[str]:
        self.open()
        self.root.destroy()
        return self.selected_button
