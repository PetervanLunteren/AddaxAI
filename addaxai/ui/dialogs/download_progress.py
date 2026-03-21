"""Download progress windows for AddaxAI."""

from typing import Any, Callable, Optional

try:
    import customtkinter
    _CTkToplevel = customtkinter.CTkToplevel
    _CTkFrame = customtkinter.CTkFrame
    _CTkLabel = customtkinter.CTkLabel
    _CTkFont = customtkinter.CTkFont
    _CTkProgressBar = customtkinter.CTkProgressBar
except ImportError:
    class _CTkToplevel:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class _CTkFrame:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def grid(self, **kwargs):
            pass
        def columnconfigure(self, *args, **kwargs):
            pass
    class _CTkLabel:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def grid(self, **kwargs):
            pass
        def configure(self, **kwargs):
            pass
    class _CTkFont:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class _CTkProgressBar:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args):
            pass
        def grid(self, **kwargs):
            pass

from addaxai.i18n import lang_idx as i18n_lang_idx
from addaxai.ui.widgets.buttons import CancelButton


class EnvDownloadProgressWindow:
    def __init__(self, env_title: str, total_size_str: str, master: Any = None,
                 scale_factor: float = 1.0, padx: int = 5, pady: int = 2,
                 green_primary: str = "#00A86B",
                 open_nosleep_func: Optional[Callable[[], None]] = None) -> None:
        self.green_primary = green_primary
        self.dm_root = _CTkToplevel(master)
        self.dm_root.title("Download progress")
        self.dm_root.geometry("+10+10")
        self.frm = _CTkFrame(master=self.dm_root)
        self.frm.grid(row=3, column=0, padx=padx, pady=(pady, pady / 2), sticky="nswe")
        self.frm.columnconfigure(0, weight=1, minsize=500 * scale_factor)

        self.lbl = _CTkLabel(self.dm_root,
                             text=[f"Downloading environment '{env_title}' ({total_size_str})",
                                   f"Descargar entorno '{env_title}' ({total_size_str})",
                                   f"Téléchargement de l'environnement '{env_title}' ({total_size_str})"][i18n_lang_idx()],
                             font=_CTkFont(family='CTkFont', size=14, weight='bold'))
        self.lbl.grid(row=0, column=0, padx=padx, pady=(0, 0), sticky="nsew")

        self.war = _CTkLabel(self.dm_root,
                             text=["Please prevent computer from sleeping during the download.",
                                   "Por favor, evite que el ordenador se duerma durante la descarga.",
                                   "SVP empêcher l'ordinateur de tomber en mode veille lors du téléchargement."][i18n_lang_idx()])
        self.war.grid(row=1, column=0, padx=padx, pady=0, sticky="nswe")

        self.but = CancelButton(self.dm_root,
                                text=["  Prevent sleep with online tool ",
                                      "  Usar prevención de sueño en línea  ",
                                      "  Prévenir la mise en veille avec un outil en ligne "][i18n_lang_idx()],
                                command=open_nosleep_func)
        self.but.grid(row=2, column=0, padx=padx, pady=(pady / 2, 0), sticky="")

        # Label for Downloading Progress
        self.lbl_download = _CTkLabel(self.frm,
                                      text=["Downloading...", "Descargando...", "Téléchargement..."][i18n_lang_idx()])
        self.lbl_download.grid(row=1, column=0, padx=padx, pady=(0, 0), sticky="nsew")

        # Progress bar for downloading
        self.pbr_download = _CTkProgressBar(self.frm, orientation="horizontal", height=22, corner_radius=5, width=1)
        self.pbr_download.set(0)
        self.pbr_download.grid(row=2, column=0, padx=padx, pady=pady, sticky="nsew")

        self.per_download = _CTkLabel(self.frm, text=f" 0% ", height=5,
                                      fg_color=("#949BA2", "#4B4D50"), text_color="white")
        self.per_download.grid(row=2, column=0, padx=padx, pady=pady, sticky="")

        # Label for Extraction Progress
        self.lbl_extraction = _CTkLabel(self.frm,
                                        text=["Extracting...", "Extrayendo...", "Extraction..."][i18n_lang_idx()])
        self.lbl_extraction.grid(row=3, column=0, padx=padx, pady=(0, 0), sticky="nsew")

        # Progress bar for extraction
        self.pbr_extraction = _CTkProgressBar(self.frm, orientation="horizontal", height=22, corner_radius=5, width=1)
        self.pbr_extraction.set(0)
        self.pbr_extraction.grid(row=4, column=0, padx=padx, pady=pady, sticky="nsew")

        self.per_extraction = _CTkLabel(self.frm, text=f" 0% ", height=5,
                                        fg_color=("#949BA2", "#4B4D50"), text_color="white")
        self.per_extraction.grid(row=4, column=0, padx=padx, pady=pady, sticky="")

        self.dm_root.withdraw()

    def open(self) -> None:
        self.dm_root.update()
        self.dm_root.deiconify()

    def update_download_progress(self, percentage: float) -> None:
        self.pbr_download.set(percentage)
        self.per_download.configure(text=f" {round(percentage * 100)}% ")
        if percentage > 0.5:
            self.per_download.configure(fg_color=(self.green_primary, "#1F6BA5"))
        else:
            self.per_download.configure(fg_color=("#949BA2", "#4B4D50"))
        self.dm_root.update()

    def update_extraction_progress(self, percentage: float) -> None:
        self.pbr_extraction.set(percentage)
        self.per_extraction.configure(text=f" {round(percentage * 100)}% ")
        if percentage > 0.5:
            self.per_extraction.configure(fg_color=(self.green_primary, "#1F6BA5"))
        else:
            self.per_extraction.configure(fg_color=("#949BA2", "#4B4D50"))
        self.dm_root.update()

    def close(self) -> None:
        self.dm_root.destroy()


class ModelDownloadProgressWindow:
    def __init__(self, model_title: str, total_size_str: str, master: Any = None,
                 scale_factor: float = 1.0, padx: int = 5, pady: int = 2,
                 green_primary: str = "#00A86B",
                 open_nosleep_func: Optional[Callable[[], None]] = None) -> None:
        self.green_primary = green_primary
        self.dm_root = _CTkToplevel(master)
        self.dm_root.title("Download progress")
        self.dm_root.geometry("+10+10")
        self.frm = _CTkFrame(master=self.dm_root)
        self.frm.grid(row=3, column=0, padx=padx, pady=(pady, pady / 2), sticky="nswe")
        self.frm.columnconfigure(0, weight=1, minsize=500 * scale_factor)

        self.lbl = _CTkLabel(self.dm_root,
                             text=[f"Downloading model '{model_title}' ({total_size_str})",
                                   f"Descargar modelo '{model_title}' ({total_size_str})",
                                   f"Téléchargement du modèle '{model_title}' ({total_size_str})"][i18n_lang_idx()],
                             font=_CTkFont(family='CTkFont', size=14, weight='bold'))
        self.lbl.grid(row=0, column=0, padx=padx, pady=(0, 0), sticky="nsew")

        self.war = _CTkLabel(self.dm_root,
                             text=["Please prevent computer from sleeping during the download.",
                                   "Por favor, evite que el ordenador se duerma durante la descarga.",
                                   "SVP empêcher l'ordinateur de tomber en mode veille pendant le téléchargement."][i18n_lang_idx()])
        self.war.grid(row=1, column=0, padx=padx, pady=0, sticky="nswe")

        self.but = CancelButton(self.dm_root,
                                text=["  Prevent sleep with online tool ",
                                      "  Usar prevención de sueño en línea  ",
                                      "  Prévenir la mise en veille avec un outil en ligne "][i18n_lang_idx()],
                                command=open_nosleep_func)
        self.but.grid(row=2, column=0, padx=padx, pady=(pady / 2, 0), sticky="")

        self.pbr = _CTkProgressBar(self.frm, orientation="horizontal", height=22, corner_radius=5, width=1)
        self.pbr.set(0)
        self.pbr.grid(row=1, column=0, padx=padx, pady=pady, sticky="nsew")

        self.per = _CTkLabel(self.frm, text=f" 0% ", height=5,
                             fg_color=("#949BA2", "#4B4D50"), text_color="white")
        self.per.grid(row=1, column=0, padx=padx, pady=pady, sticky="")

        self.dm_root.withdraw()

    def open(self) -> None:
        self.dm_root.update()
        self.dm_root.deiconify()

    def update_progress(self, percentage: float) -> None:
        self.pbr.set(percentage)
        self.per.configure(text=f" {round(percentage * 100)}% ")
        if percentage > 0.5:
            self.per.configure(fg_color=(self.green_primary, "#1F6BA5"))
        else:
            self.per.configure(fg_color=("#949BA2", "#4B4D50"))
        self.dm_root.update()

    def close(self) -> None:
        self.dm_root.destroy()
