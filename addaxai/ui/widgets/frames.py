"""Reusable CTkFrame subclasses for AddaxAI."""

from typing import Any

try:
    import customtkinter
    _CTkFrame = customtkinter.CTkFrame
except ImportError:
    class _CTkFrame:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def columnconfigure(self, *args, **kwargs):
            pass


class MyMainFrame(_CTkFrame):
    def __init__(self, master: Any, scale_factor: float = 1.0, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        if scale_factor != 1.0:
            self.columnconfigure(0, weight=1, minsize=70 * round(scale_factor * 1.35, 2))
            self.columnconfigure(1, weight=1, minsize=350 * round(scale_factor * 1.35, 2))
        else:
            self.columnconfigure(0, weight=1, minsize=70)
            self.columnconfigure(1, weight=1, minsize=350)


class MySubFrame(_CTkFrame):
    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1, minsize=250)
        self.columnconfigure(1, weight=1, minsize=250)


class MySubSubFrame(_CTkFrame):
    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
