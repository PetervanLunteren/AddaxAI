"""Info frame classes for AddaxAI dialogs."""

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


class ModelInfoFrame(_CTkFrame):
    def __init__(self, master: Any, scale_factor: float = 1.0, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1, minsize=120 * scale_factor)
        self.columnconfigure(1, weight=1, minsize=500 * scale_factor)


class DonationPopupFrame(_CTkFrame):
    def __init__(self, master: Any, scale_factor: float = 1.0, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1, minsize=500 * scale_factor)
