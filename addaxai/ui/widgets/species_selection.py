"""Species selection scrollable checkbox frame for AddaxAI."""

from typing import Any, Callable, List, Optional

try:
    import customtkinter
    _CTkScrollableFrame = customtkinter.CTkScrollableFrame
    _CTkCheckBox = customtkinter.CTkCheckBox
except ImportError:
    class _CTkScrollableFrame:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class _CTkCheckBox:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

from addaxai.i18n import lang_idx as i18n_lang_idx


class SpeciesSelectionFrame(_CTkScrollableFrame):
    def __init__(self, master: Any, all_classes: List[str] = [], selected_classes: List[str] = [],
                 command: Optional[Callable[[], None]] = None, dummy_spp: bool = False,
                 pady: int = 2, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.dummy_spp = dummy_spp
        if dummy_spp:
            all_classes = [f"{['Species', 'Especies', 'Espèces'][i18n_lang_idx()]} {i + 1}" for i in range(10)]
        self.command = command
        self.checkbox_list = []
        self.selected_classes = selected_classes
        for item in all_classes:
            self.add_item(item, pady)

    def add_item(self, item: str, pady: int = 2) -> None:
        checkbox = _CTkCheckBox(self, text=item)
        if self.dummy_spp:
            checkbox.configure(state="disabled")
        if item in self.selected_classes:
            checkbox.select()
        if self.command is not None:
            checkbox.configure(command=self.command)
        checkbox.grid(row=len(self.checkbox_list), column=0, pady=pady, sticky="nsw")
        self.checkbox_list.append(checkbox)

    def get_checked_items(self) -> List[str]:
        return [checkbox.cget('text') for checkbox in self.checkbox_list if checkbox.get() == 1]

    def get_all_items(self) -> List[str]:
        return [checkbox.cget('text') for checkbox in self.checkbox_list]
