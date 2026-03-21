"""Reusable CTkButton subclasses for AddaxAI."""

from typing import Any

try:
    import customtkinter
    _CTkButton = customtkinter.CTkButton
except ImportError:
    class _CTkButton:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def configure(self, **kwargs):
            pass


class InfoButton(_CTkButton):  # type: ignore[misc, valid-type]
    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.configure(fg_color=("#ebebeb", "#333333"),
                       hover=False,
                       text_color=("grey", "grey"),
                       height=1,
                       width=1)


class CancelButton(_CTkButton):  # type: ignore[misc, valid-type]
    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.configure(fg_color=("#ebeaea", "#4B4D50"),
                       hover_color=("#939aa2", "#2B2B2B"),
                       text_color=("black", "white"),
                       height=10,
                       width=120)


class GreyTopButton(_CTkButton):  # type: ignore[misc, valid-type]
    def __init__(self, master: Any, yellow_secondary: str = "#F0EEDC",
                 yellow_tertiary: str = "#E4E1D0", border_width: int = 0, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.configure(fg_color=(yellow_secondary, "#333333"),
                       hover_color=(yellow_tertiary, "#2B2B2B"),
                       text_color=("black", "white"),
                       height=10,
                       width=140,
                       border_width=border_width)
