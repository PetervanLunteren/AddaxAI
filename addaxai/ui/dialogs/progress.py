"""ProgressWindow — deploy and postprocess progress dialog for AddaxAI."""

from typing import Any, Callable, List

try:
    import customtkinter
    _CTkToplevel = customtkinter.CTkToplevel
    _CTkFont = customtkinter.CTkFont
    _CTkFrame = customtkinter.CTkFrame
    _CTkLabel = customtkinter.CTkLabel
    _CTkProgressBar = customtkinter.CTkProgressBar
except ImportError:
    class _CTkToplevel:
        def __init__(self, *a, **kw): pass
        def title(self, *a): pass
        def geometry(self, *a): pass
        def update(self): pass
        def deiconify(self): pass
        def withdraw(self): pass
        def destroy(self): pass
        def protocol(self, *a, **kw): pass
    class _CTkFont:
        def __init__(self, *a, **kw): pass
    class _CTkFrame:
        def __init__(self, *a, **kw): pass
        def grid(self, **kw): pass
        def columnconfigure(self, *a, **kw): pass
        def grid_slaves(self, *a, **kw): return []
    class _CTkLabel:
        def __init__(self, *a, **kw): pass
        def grid(self, **kw): pass
        def grid_remove(self): pass
        def configure(self, **kw): pass
    class _CTkProgressBar:
        def __init__(self, *a, **kw): pass
        def set(self, v): pass
        def grid(self, **kw): pass

from addaxai.i18n import t, lang_idx as i18n_lang_idx
from addaxai.ui.widgets.buttons import CancelButton

class ProgressWindow:
    def __init__(self, processes: List[str], master: Any = None, scale_factor: float = 1.0,
                 padx: int = 5, pady: int = 2, green_primary: str = "#00A86B") -> None:
        self.progress_top_level_window = _CTkToplevel(master)
        self.progress_top_level_window.title(t('analysis_progress'))
        self.progress_top_level_window.geometry("+10+10")
        lbl_height = 12
        pbr_height = 22
        ttl_font = _CTkFont(family='CTkFont', size=13, weight = 'bold')
        self.pady_progress_window = pady/1.5
        self.padx_progress_window = padx/1.5
        self.green_primary = green_primary
        
        # language settings
        in_queue_txt = ['In queue', 'En cola', 'Mise en file d\'attente']
        checking_fpaths_txt = ['Checking file paths', 'Comprobación de rutas de archivos', 'Vérification des chemins de fichiers']
        processing_image_txt = ['Processing image', 'Procesamiento de imágenes', 'Traitement des images']
        processing_animal_txt = ['Processing animal', 'Procesamiento de animales', 'Traitement des animaux']
        processing_unknown_txt = ['Processing', 'Procesamiento', 'Traitement']
        images_per_second_txt = ['Images per second', 'Imágenes por segundo', 'Images par seconde']
        animals_per_second_txt = ['Animals per second', 'Animales por segundo', 'Animaux par seconde']
        frames_per_second_txt = ['Frames per second', 'Fotogramas por segundo', 'Trames par seconde']
        elapsed_time_txt = ["Elapsed time", "Tiempo transcurrido", "Temps écoulé"]
        remaining_time_txt = ["Remaining time", "Tiempo restante", "Temps restant"]
        running_on_txt = ["Running on", "Funcionando en", "S'exécute sur"]

        # clarify titles if both images and videos are being processed
        if "img_det" in processes and "vid_det" in processes:
            img_det_extra_string = t('in_images')
            vid_det_extra_string = t('in_videos')
        else:
            img_det_extra_string = ""
            vid_det_extra_string = ""
        if "img_pst" in processes and "vid_pst" in processes:
            img_pst_extra_string = t('images_pst')
            vid_pst_extra_string = t('videos_pst')
        else:
            img_pst_extra_string = ""
            vid_pst_extra_string = ""

        # initialise image detection process
        if "img_det" in processes:
            self.img_det_frm = _CTkFrame(master=self.progress_top_level_window)
            self.img_det_frm.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            img_det_ttl_txt = [f'Locating animals{img_det_extra_string}...', f'Localización de animales{img_det_extra_string}...', f'Localisation des animaux{img_det_extra_string}...']
            self.img_det_ttl = _CTkLabel(self.img_det_frm, text=img_det_ttl_txt[i18n_lang_idx()], font = ttl_font)
            self.img_det_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.img_det_sub_frm = _CTkFrame(master=self.img_det_frm)
            self.img_det_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.img_det_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.img_det_pbr = _CTkProgressBar(self.img_det_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.img_det_pbr.set(0)
            self.img_det_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.img_det_per = _CTkLabel(self.img_det_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.img_det_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.img_det_wai_lbl = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=checking_fpaths_txt[i18n_lang_idx()])
            self.img_det_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.img_det_num_lbl = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{processing_image_txt[i18n_lang_idx()]}:")
            self.img_det_num_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_num_lbl.grid_remove()
            self.img_det_num_val = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_num_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_det_num_val.grid_remove()
            self.img_det_ela_lbl = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[i18n_lang_idx()]}:")
            self.img_det_ela_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_ela_lbl.grid_remove()
            self.img_det_ela_val = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_ela_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.img_det_ela_val.grid_remove()
            self.img_det_rem_lbl = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{remaining_time_txt[i18n_lang_idx()]}:")
            self.img_det_rem_lbl.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_rem_lbl.grid_remove()
            self.img_det_rem_val = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_rem_val.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_det_rem_val.grid_remove()
            self.img_det_spe_lbl = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{images_per_second_txt[i18n_lang_idx()]}:")
            self.img_det_spe_lbl.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_spe_lbl.grid_remove()
            self.img_det_spe_val = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_spe_val.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_det_spe_val.grid_remove()
            self.img_det_hwa_lbl = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{running_on_txt[i18n_lang_idx()]}:")
            self.img_det_hwa_lbl.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_hwa_lbl.grid_remove()
            self.img_det_hwa_val = _CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_hwa_val.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_det_hwa_val.grid_remove()
            self.img_det_can_btn = CancelButton(master = self.img_det_sub_frm, text = "Cancel", command = lambda: print(""))
            self.img_det_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.img_det_can_btn.grid_remove()

        # initialise image classification process
        if "img_cls" in processes:
            self.img_cls_frm = _CTkFrame(master=self.progress_top_level_window)
            self.img_cls_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            img_cls_ttl_txt = [f'Identifying animals{img_det_extra_string}...', f'Identificación de animales{img_det_extra_string}...', f'Identification des animaux{img_det_extra_string}...']
            self.img_cls_ttl = _CTkLabel(self.img_cls_frm, text=img_cls_ttl_txt[i18n_lang_idx()], font = ttl_font)
            self.img_cls_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.img_cls_sub_frm = _CTkFrame(master=self.img_cls_frm)
            self.img_cls_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.img_cls_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.img_cls_pbr = _CTkProgressBar(self.img_cls_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.img_cls_pbr.set(0)
            self.img_cls_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.img_cls_per = _CTkLabel(self.img_cls_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.img_cls_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.img_cls_wai_lbl = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=in_queue_txt[i18n_lang_idx()])
            self.img_cls_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.img_cls_num_lbl = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{processing_animal_txt[i18n_lang_idx()]}:")
            self.img_cls_num_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_num_lbl.grid_remove()
            self.img_cls_num_val = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_num_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_cls_num_val.grid_remove()
            self.img_cls_ela_lbl = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[i18n_lang_idx()]}:")
            self.img_cls_ela_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_ela_lbl.grid_remove()
            self.img_cls_ela_val = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_ela_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.img_cls_ela_val.grid_remove()
            self.img_cls_rem_lbl = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{remaining_time_txt[i18n_lang_idx()]}:")
            self.img_cls_rem_lbl.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_rem_lbl.grid_remove()
            self.img_cls_rem_val = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_rem_val.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_cls_rem_val.grid_remove()
            self.img_cls_spe_lbl = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{animals_per_second_txt[i18n_lang_idx()]}:")
            self.img_cls_spe_lbl.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_spe_lbl.grid_remove()
            self.img_cls_spe_val = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_spe_val.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_cls_spe_val.grid_remove()
            self.img_cls_hwa_lbl = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{running_on_txt[i18n_lang_idx()]}:")
            self.img_cls_hwa_lbl.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_hwa_lbl.grid_remove()
            self.img_cls_hwa_val = _CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_hwa_val.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_cls_hwa_val.grid_remove()
            self.img_cls_can_btn = CancelButton(master = self.img_cls_sub_frm, text = "Cancel", command = lambda: print(""))
            self.img_cls_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.img_cls_can_btn.grid_remove()

        # initialise video detection process
        if "vid_det" in processes:
            self.vid_det_frm = _CTkFrame(master=self.progress_top_level_window)
            self.vid_det_frm.grid(row=2, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            vid_det_ttl_txt = [f'Locating animals{vid_det_extra_string}...', f'Localización de animales{vid_det_extra_string}...', f'Localisation des animaux{vid_det_extra_string}...']
            self.vid_det_ttl = _CTkLabel(self.vid_det_frm, text=vid_det_ttl_txt[i18n_lang_idx()], font = ttl_font)
            self.vid_det_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.vid_det_sub_frm = _CTkFrame(master=self.vid_det_frm)
            self.vid_det_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.vid_det_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.vid_det_pbr = _CTkProgressBar(self.vid_det_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.vid_det_pbr.set(0)
            self.vid_det_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.vid_det_per = _CTkLabel(self.vid_det_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.vid_det_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.vid_det_wai_lbl = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=in_queue_txt[i18n_lang_idx()])
            self.vid_det_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.vid_det_num_lbl = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{processing_unknown_txt[i18n_lang_idx()]}:")
            self.vid_det_num_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_num_lbl.grid_remove()
            self.vid_det_num_val = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_num_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_det_num_val.grid_remove()
            self.vid_det_ela_lbl = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[i18n_lang_idx()]}:")
            self.vid_det_ela_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_ela_lbl.grid_remove()
            self.vid_det_ela_val = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_ela_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.vid_det_ela_val.grid_remove()
            self.vid_det_rem_lbl = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{remaining_time_txt[i18n_lang_idx()]}:")
            self.vid_det_rem_lbl.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_rem_lbl.grid_remove()
            self.vid_det_rem_val = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_rem_val.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_det_rem_val.grid_remove()
            self.vid_det_spe_lbl = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{frames_per_second_txt[i18n_lang_idx()]}:")
            self.vid_det_spe_lbl.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_spe_lbl.grid_remove()
            self.vid_det_spe_val = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_spe_val.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_det_spe_val.grid_remove()
            self.vid_det_hwa_lbl = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{running_on_txt[i18n_lang_idx()]}:")
            self.vid_det_hwa_lbl.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_hwa_lbl.grid_remove()
            self.vid_det_hwa_val = _CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_hwa_val.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_det_hwa_val.grid_remove()
            self.vid_det_can_btn = CancelButton(master = self.vid_det_sub_frm, text = "Cancel", command = lambda: print(""))
            self.vid_det_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.vid_det_can_btn.grid_remove()

        # initialise video classification process
        if "vid_cls" in processes:
            self.vid_cls_frm = _CTkFrame(master=self.progress_top_level_window)
            self.vid_cls_frm.grid(row=3, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            vid_cls_ttl_txt = [f'Identifying animals{vid_det_extra_string}...', f'Identificación de animales{vid_det_extra_string}...', f'Identification des animaux{vid_det_extra_string}...']
            self.vid_cls_ttl = _CTkLabel(self.vid_cls_frm, text=vid_cls_ttl_txt[i18n_lang_idx()], font = ttl_font)
            self.vid_cls_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.vid_cls_sub_frm = _CTkFrame(master=self.vid_cls_frm)
            self.vid_cls_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.vid_cls_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.vid_cls_pbr = _CTkProgressBar(self.vid_cls_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.vid_cls_pbr.set(0)
            self.vid_cls_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.vid_cls_per = _CTkLabel(self.vid_cls_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.vid_cls_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.vid_cls_wai_lbl = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=in_queue_txt[i18n_lang_idx()])
            self.vid_cls_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.vid_cls_num_lbl = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{processing_animal_txt[i18n_lang_idx()]}:")
            self.vid_cls_num_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_num_lbl.grid_remove()
            self.vid_cls_num_val = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_num_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_cls_num_val.grid_remove()
            self.vid_cls_ela_lbl = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[i18n_lang_idx()]}:")
            self.vid_cls_ela_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_ela_lbl.grid_remove()
            self.vid_cls_ela_val = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_ela_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.vid_cls_ela_val.grid_remove()
            self.vid_cls_rem_lbl = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{remaining_time_txt[i18n_lang_idx()]}:")
            self.vid_cls_rem_lbl.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_rem_lbl.grid_remove()
            self.vid_cls_rem_val = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_rem_val.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_cls_rem_val.grid_remove()
            self.vid_cls_spe_lbl = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{animals_per_second_txt[i18n_lang_idx()]}:")
            self.vid_cls_spe_lbl.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_spe_lbl.grid_remove()
            self.vid_cls_spe_val = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_spe_val.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_cls_spe_val.grid_remove()
            self.vid_cls_hwa_lbl = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{running_on_txt[i18n_lang_idx()]}:")
            self.vid_cls_hwa_lbl.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_hwa_lbl.grid_remove()
            self.vid_cls_hwa_val = _CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_hwa_val.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_cls_hwa_val.grid_remove()
            self.vid_cls_can_btn = CancelButton(master = self.vid_cls_sub_frm, text = "Cancel", command = lambda: print(""))
            self.vid_cls_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.vid_cls_can_btn.grid_remove()

        # postprocessing for images
        if "img_pst" in processes:
            self.img_pst_frm = _CTkFrame(master=self.progress_top_level_window)
            self.img_pst_frm.grid(row=4, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            img_pst_ttl_txt = [f'Postprocessing{img_pst_extra_string}...', f'Postprocesado{img_pst_extra_string}...', f'Post-traitement des{img_pst_extra_string}...']
            self.img_pst_ttl = _CTkLabel(self.img_pst_frm, text=img_pst_ttl_txt[i18n_lang_idx()], font = ttl_font)
            self.img_pst_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.img_pst_sub_frm = _CTkFrame(master=self.img_pst_frm)
            self.img_pst_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.img_pst_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.img_pst_pbr = _CTkProgressBar(self.img_pst_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.img_pst_pbr.set(0)
            self.img_pst_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.img_pst_per = _CTkLabel(self.img_pst_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.img_pst_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.img_pst_wai_lbl = _CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=in_queue_txt[i18n_lang_idx()])
            self.img_pst_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.img_pst_ela_lbl = _CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[i18n_lang_idx()]}:")
            self.img_pst_ela_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_pst_ela_lbl.grid_remove()
            self.img_pst_ela_val = _CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=f"")
            self.img_pst_ela_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.img_pst_ela_val.grid_remove()
            self.img_pst_rem_lbl = _CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=f"{remaining_time_txt[i18n_lang_idx()]}:")
            self.img_pst_rem_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_pst_rem_lbl.grid_remove()
            self.img_pst_rem_val = _CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=f"")
            self.img_pst_rem_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.img_pst_rem_val.grid_remove()
            self.img_pst_can_btn = CancelButton(master = self.img_pst_sub_frm, text = "Cancel", command = lambda: print(""))
            self.img_pst_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.img_pst_can_btn.grid_remove()

        # postprocessing for videos
        if "vid_pst" in processes:
            self.vid_pst_frm = _CTkFrame(master=self.progress_top_level_window)
            self.vid_pst_frm.grid(row=5, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            vid_pst_ttl_txt = [f'Postprocessing{vid_pst_extra_string}...', f'Postprocesado{vid_pst_extra_string}...', f'Post-traitement des{vid_pst_extra_string}...']
            self.vid_pst_ttl = _CTkLabel(self.vid_pst_frm, text=vid_pst_ttl_txt[i18n_lang_idx()], font = ttl_font)
            self.vid_pst_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.vid_pst_sub_frm = _CTkFrame(master=self.vid_pst_frm)
            self.vid_pst_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.vid_pst_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.vid_pst_pbr = _CTkProgressBar(self.vid_pst_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.vid_pst_pbr.set(0)
            self.vid_pst_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.vid_pst_per = _CTkLabel(self.vid_pst_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.vid_pst_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.vid_pst_wai_lbl = _CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=in_queue_txt[i18n_lang_idx()])
            self.vid_pst_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.vid_pst_ela_lbl = _CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[i18n_lang_idx()]}:")
            self.vid_pst_ela_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_pst_ela_lbl.grid_remove()
            self.vid_pst_ela_val = _CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=f"")
            self.vid_pst_ela_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.vid_pst_ela_val.grid_remove()
            self.vid_pst_rem_lbl = _CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=f"{remaining_time_txt[i18n_lang_idx()]}:")
            self.vid_pst_rem_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_pst_rem_lbl.grid_remove()
            self.vid_pst_rem_val = _CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=f"")
            self.vid_pst_rem_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.vid_pst_rem_val.grid_remove()
            self.vid_pst_can_btn = CancelButton(master = self.vid_pst_sub_frm, text = "Cancel", command = lambda: print(""))
            self.vid_pst_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.vid_pst_can_btn.grid_remove()

        # plotting can only be done for images
        if "plt" in processes:
            self.plt_frm = _CTkFrame(master=self.progress_top_level_window)
            self.plt_frm.grid(row=6, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            plt_ttl_txt = [f'Creating graphs...', f'Creando gráficos...', f'Création des graphiques...']
            self.plt_ttl = _CTkLabel(self.plt_frm, text=plt_ttl_txt[i18n_lang_idx()], font = ttl_font)
            self.plt_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.plt_sub_frm = _CTkFrame(master=self.plt_frm)
            self.plt_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.plt_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.plt_pbr = _CTkProgressBar(self.plt_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.plt_pbr.set(0)
            self.plt_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.plt_per = _CTkLabel(self.plt_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.plt_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.plt_wai_lbl = _CTkLabel(self.plt_sub_frm, height = lbl_height, text=in_queue_txt[i18n_lang_idx()])
            self.plt_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.plt_ela_lbl = _CTkLabel(self.plt_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[i18n_lang_idx()]}:")
            self.plt_ela_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.plt_ela_lbl.grid_remove()
            self.plt_ela_val = _CTkLabel(self.plt_sub_frm, height = lbl_height, text=f"")
            self.plt_ela_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.plt_ela_val.grid_remove()
            self.plt_rem_lbl = _CTkLabel(self.plt_sub_frm, height = lbl_height, text=f"{remaining_time_txt[i18n_lang_idx()]}:")
            self.plt_rem_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.plt_rem_lbl.grid_remove()
            self.plt_rem_val = _CTkLabel(self.plt_sub_frm, height = lbl_height, text=f"")
            self.plt_rem_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.plt_rem_val.grid_remove()
            self.plt_can_btn = CancelButton(master = self.plt_sub_frm, text = "Cancel", command = lambda: print(""))
            self.plt_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.plt_can_btn.grid_remove()

        self.progress_top_level_window.update()

    def update_values(self,
                      process: str,
                      status: str,
                      cur_it: int = 1,
                      tot_it: int = 1,
                      time_ela: str = "",
                      time_rem: str = "",
                      speed: str = "",
                      hware: str = "",
                      cancel_func: Callable[[], None] = lambda: None,
                      extracting_frames_txt: List[str] = ["Extracting frames...     ",
                                                          "Extrayendo fotogramas...     ",
                                                          "Extraction des trames..."],
                      frame_video_choice: str = "frame") -> None:

        # language settings
        algorithm_starting_txt = ["Algorithm is starting up...", 'El algoritmo está arrancando...', 'Algorithme en démarrage...']
        smoothing_txt = ["Smoothing predictions...", 'Suavizar las predicciones...', 'Lissage de prédictions...']
        image_per_second_txt = ["Images per second:", "Imágenes por segundo:", "Images par seconde:"]
        seconds_per_image_txt = ["Seconds per image:", "Segundos por imagen:", "Secondes par image:"]
        animals_per_second_txt = ["Animals per second:", "Animales por segundo:", "Animaux par seconde:"]
        seconds_per_animal_txt = ["Seconds per animal:", "Segundos por animal:", "Secondes par animal:"]
        frames_per_second_txt = ["Frames per second:", "Fotogramas por segundo:"], "Trames par seconde:"
        seconds_per_frame_txt = ["Seconds per frame:", "Segundos por fotograma:", "Secondes par trame:"]
        videos_per_second_txt = ["Videos per second:", "Vídeos por segundo:", "Vidéos par seconde:"]
        seconds_per_video_txt = ["Seconds per video:", "Segundos por vídeo:", "Secondes par vidéo:"]
        processing_videos_txt = ["Processing video:", "Procesando vídeo:", "Traitement du vidéo:"]
        processing_frames_txt = ["Processing frame:", "Procesando fotograma:", "Traitement de la trame:"]
        starting_up_txt = ["Starting up...", "Arrancando...", "Démarrage..."]

        # detection of images
        if process == "img_det":
            if status == "load":
                self.img_det_wai_lbl.configure(text = algorithm_starting_txt[i18n_lang_idx()])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.img_det_wai_lbl.grid_remove()
                    self.img_det_num_lbl.grid()
                    self.img_det_num_val.grid()
                    self.img_det_ela_lbl.grid()
                    self.img_det_ela_val.grid()
                    self.img_det_rem_lbl.grid()
                    self.img_det_rem_val.grid()
                    self.img_det_spe_lbl.grid()
                    self.img_det_spe_val.grid()
                    self.img_det_hwa_lbl.grid()
                    self.img_det_hwa_val.grid()
                    self.img_det_can_btn.grid()
                    self.img_det_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.img_det_pbr.set(percentage)
                self.img_det_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.img_det_per.configure(fg_color=(self.green_primary, "#1F6BA5"))
                else:
                    self.img_det_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.img_det_num_val.configure(text = f"{cur_it} of {tot_it}")
                self.img_det_ela_val.configure(text = time_ela)
                self.img_det_rem_val.configure(text = time_rem)
                self.img_det_spe_lbl.configure(text = image_per_second_txt[i18n_lang_idx()] if "it/s" in speed else seconds_per_image_txt[i18n_lang_idx()])
                parsed_speed = speed.replace("it/s", "").replace("s/it", "")
                self.img_det_spe_val.configure(text = parsed_speed)
                self.img_det_hwa_val.configure(text = hware)
            elif status == "done":
                self.img_det_num_lbl.grid_remove()
                self.img_det_num_val.grid_remove()
                self.img_det_rem_lbl.grid_remove()
                self.img_det_rem_val.grid_remove()
                self.img_det_hwa_lbl.grid_remove()
                self.img_det_hwa_val.grid_remove()
                self.img_det_can_btn.grid_remove()
                self.img_det_ela_val.grid_remove()
                self.img_det_ela_lbl.grid_remove()
                self.img_det_spe_lbl.grid_remove()
                self.img_det_spe_val.grid_remove()
                self.img_det_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.img_det_per.grid_configure(pady=(self.pady_progress_window, 0))
                
        # classification of images
        elif process == "img_cls":
            if status == "load":
                self.img_cls_wai_lbl.configure(text = algorithm_starting_txt[i18n_lang_idx()])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.img_cls_wai_lbl.grid_remove()
                    self.img_cls_num_lbl.grid()
                    self.img_cls_num_val.grid()
                    self.img_cls_ela_lbl.grid()
                    self.img_cls_ela_val.grid()
                    self.img_cls_rem_lbl.grid()
                    self.img_cls_rem_val.grid()
                    self.img_cls_spe_lbl.grid()
                    self.img_cls_spe_val.grid()
                    self.img_cls_hwa_lbl.grid()
                    self.img_cls_hwa_val.grid()
                    self.img_cls_can_btn.grid()
                    self.img_cls_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.img_cls_pbr.set(percentage)
                self.img_cls_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.img_cls_per.configure(fg_color=(self.green_primary, "#1F6BA5"))
                else:
                    self.img_cls_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.img_cls_num_val.configure(text = f"{cur_it} of {tot_it}")
                self.img_cls_ela_val.configure(text = time_ela)
                self.img_cls_rem_val.configure(text = time_rem)
                self.img_cls_spe_lbl.configure(text = animals_per_second_txt[i18n_lang_idx()] if "it/s" in speed else seconds_per_animal_txt[i18n_lang_idx()])
                parsed_speed = speed.replace("it/s", "").replace("s/it", "")
                self.img_cls_spe_val.configure(text = parsed_speed)
                self.img_cls_hwa_val.configure(text = hware)
            elif status == "smoothing":
                self.img_cls_num_lbl.grid_remove()
                self.img_cls_num_val.grid_remove()
                self.img_cls_rem_lbl.grid_remove()
                self.img_cls_rem_val.grid_remove()
                self.img_cls_hwa_lbl.grid_remove()
                self.img_cls_hwa_val.grid_remove()
                self.img_cls_can_btn.grid_remove()
                self.img_cls_ela_val.grid_remove()
                self.img_cls_ela_lbl.grid_remove()
                self.img_cls_spe_lbl.grid_remove()
                self.img_cls_spe_val.grid_remove()
                self.img_cls_wai_lbl.grid()
                self.img_cls_wai_lbl.configure(text = smoothing_txt[i18n_lang_idx()])
            elif status == "done":
                self.img_cls_num_lbl.grid_remove()
                self.img_cls_num_val.grid_remove()
                self.img_cls_rem_lbl.grid_remove()
                self.img_cls_rem_val.grid_remove()
                self.img_cls_hwa_lbl.grid_remove()
                self.img_cls_hwa_val.grid_remove()
                self.img_cls_can_btn.grid_remove()
                self.img_cls_ela_val.grid_remove()
                self.img_cls_ela_lbl.grid_remove()
                self.img_cls_spe_lbl.grid_remove()
                self.img_cls_spe_val.grid_remove()
                self.img_cls_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.img_cls_per.grid_configure(pady=(self.pady_progress_window, 0))

        # detection of videos
        if process == "vid_det":
            if status == "load":
                self.vid_det_wai_lbl.configure(text = algorithm_starting_txt[i18n_lang_idx()])
                self.just_shown_load_screen = True
            elif status == "extracting frames":
                self.vid_det_wai_lbl.configure(text = extracting_frames_txt[i18n_lang_idx()])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.vid_det_wai_lbl.grid_remove()
                    self.vid_det_num_lbl.grid()
                    self.vid_det_num_val.grid()
                    self.vid_det_ela_lbl.grid()
                    self.vid_det_ela_val.grid()
                    self.vid_det_rem_lbl.grid()
                    self.vid_det_rem_val.grid()
                    self.vid_det_spe_lbl.grid()
                    self.vid_det_spe_val.grid()
                    self.vid_det_hwa_lbl.grid()
                    self.vid_det_hwa_val.grid()
                    self.vid_det_can_btn.grid()
                    self.vid_det_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.vid_det_pbr.set(percentage)
                self.vid_det_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.vid_det_per.configure(fg_color=(self.green_primary, "#1F6BA5"))
                else:
                    self.vid_det_per.configure(fg_color=("#949BA2", "#4B4D50"))
                if frame_video_choice == "frame":
                    self.vid_det_num_lbl.configure(text = processing_frames_txt[i18n_lang_idx()])
                else:
                    self.vid_det_num_lbl.configure(text = processing_videos_txt[i18n_lang_idx()])
                self.vid_det_num_val.configure(text = f"{cur_it} of {tot_it}")
                self.vid_det_ela_val.configure(text = time_ela)
                self.vid_det_rem_val.configure(text = time_rem)
                if frame_video_choice == "frame":
                    self.vid_det_spe_lbl.configure(text = frames_per_second_txt[i18n_lang_idx()] if "it/s" in speed else seconds_per_frame_txt[i18n_lang_idx()])
                else:
                    self.vid_det_spe_lbl.configure(text = videos_per_second_txt[i18n_lang_idx()] if "it/s" in speed else seconds_per_video_txt[i18n_lang_idx()])
                parsed_speed = speed.replace("it/s", "").replace("s/it", "")
                self.vid_det_spe_val.configure(text = parsed_speed)
                self.vid_det_hwa_val.configure(text = hware)
            elif status == "done":
                self.vid_det_num_lbl.grid_remove()
                self.vid_det_num_val.grid_remove()
                self.vid_det_rem_lbl.grid_remove()
                self.vid_det_rem_val.grid_remove()
                self.vid_det_hwa_lbl.grid_remove()
                self.vid_det_hwa_val.grid_remove()
                self.vid_det_ela_val.grid_remove()
                self.vid_det_ela_lbl.grid_remove()
                self.vid_det_spe_lbl.grid_remove()
                self.vid_det_spe_val.grid_remove()
                self.vid_det_can_btn.grid_remove()
                self.vid_det_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.vid_det_per.grid_configure(pady=(self.pady_progress_window, 0))

        # classification of videos
        elif process == "vid_cls":
            if status == "load":
                self.vid_cls_wai_lbl.configure(text = algorithm_starting_txt[i18n_lang_idx()])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.vid_cls_wai_lbl.grid_remove()
                    self.vid_cls_num_lbl.grid()
                    self.vid_cls_num_val.grid()
                    self.vid_cls_ela_lbl.grid()
                    self.vid_cls_ela_val.grid()
                    self.vid_cls_rem_lbl.grid()
                    self.vid_cls_rem_val.grid()
                    self.vid_cls_spe_lbl.grid()
                    self.vid_cls_spe_val.grid()
                    self.vid_cls_hwa_lbl.grid()
                    self.vid_cls_hwa_val.grid()
                    self.vid_cls_can_btn.grid()
                    self.vid_cls_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.vid_cls_pbr.set(percentage)
                self.vid_cls_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.vid_cls_per.configure(fg_color=(self.green_primary, "#1F6BA5"))
                else:
                    self.vid_cls_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.vid_cls_num_val.configure(text = f"{cur_it} of {tot_it}")
                self.vid_cls_ela_val.configure(text = time_ela)
                self.vid_cls_rem_val.configure(text = time_rem)
                self.vid_cls_spe_lbl.configure(text = animals_per_second_txt[i18n_lang_idx()] if "it/s" in speed else seconds_per_animal_txt[i18n_lang_idx()])
                parsed_speed = speed.replace("it/s", "").replace("s/it", "")
                self.vid_cls_spe_val.configure(text = parsed_speed)
                self.vid_cls_hwa_val.configure(text = hware)
            elif status == "smoothing":
                self.vid_cls_num_lbl.grid_remove()
                self.vid_cls_num_val.grid_remove()
                self.vid_cls_rem_lbl.grid_remove()
                self.vid_cls_rem_val.grid_remove()
                self.vid_cls_hwa_lbl.grid_remove()
                self.vid_cls_hwa_val.grid_remove()
                self.vid_cls_can_btn.grid_remove()
                self.vid_cls_ela_val.grid_remove()
                self.vid_cls_ela_lbl.grid_remove()
                self.vid_cls_spe_lbl.grid_remove()
                self.vid_cls_spe_val.grid_remove()
                self.vid_cls_wai_lbl.grid()
                self.vid_cls_wai_lbl.configure(text = smoothing_txt[i18n_lang_idx()])
            elif status == "done":
                self.vid_cls_num_lbl.grid_remove()
                self.vid_cls_num_val.grid_remove()
                self.vid_cls_rem_lbl.grid_remove()
                self.vid_cls_rem_val.grid_remove()
                self.vid_cls_hwa_lbl.grid_remove()
                self.vid_cls_hwa_val.grid_remove()
                self.vid_cls_ela_val.grid_remove()
                self.vid_cls_ela_lbl.grid_remove()
                self.vid_cls_spe_lbl.grid_remove()
                self.vid_cls_spe_val.grid_remove()
                self.vid_cls_can_btn.grid_remove()
                self.vid_cls_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.vid_cls_per.grid_configure(pady=(self.pady_progress_window, 0))

        # postprocessing of images
        elif process == "img_pst":
            if status == "load":
                self.img_pst_wai_lbl.configure(text = starting_up_txt[i18n_lang_idx()])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.img_pst_wai_lbl.grid_remove()
                    self.img_pst_ela_lbl.grid()
                    self.img_pst_ela_val.grid()
                    self.img_pst_rem_lbl.grid()
                    self.img_pst_rem_val.grid()
                    self.img_pst_can_btn.grid()
                    self.img_pst_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.img_pst_pbr.set(percentage)
                self.img_pst_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.img_pst_per.configure(fg_color=(self.green_primary, "#1F6BA5"))
                else:
                    self.img_pst_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.img_pst_ela_val.configure(text = time_ela)
                self.img_pst_rem_val.configure(text = time_rem)
            elif status == "done":
                self.img_pst_rem_lbl.grid_remove()
                self.img_pst_rem_val.grid_remove()
                self.img_pst_ela_val.grid_remove()
                self.img_pst_ela_lbl.grid_remove()
                self.img_pst_can_btn.grid_remove()
                self.img_pst_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.img_pst_per.grid_configure(pady=(self.pady_progress_window, 0))

        # postprocessing of videos
        elif process == "vid_pst":
            if status == "load":
                self.vid_pst_wai_lbl.configure(text = starting_up_txt[i18n_lang_idx()])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.vid_pst_wai_lbl.grid_remove()
                    self.vid_pst_ela_lbl.grid()
                    self.vid_pst_ela_val.grid()
                    self.vid_pst_rem_lbl.grid()
                    self.vid_pst_rem_val.grid()
                    self.vid_pst_can_btn.grid()
                    self.vid_pst_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.vid_pst_pbr.set(percentage)
                self.vid_pst_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.vid_pst_per.configure(fg_color=(self.green_primary, "#1F6BA5"))
                else:
                    self.vid_pst_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.vid_pst_ela_val.configure(text = time_ela)
                self.vid_pst_rem_val.configure(text = time_rem)
            elif status == "done":
                self.vid_pst_rem_lbl.grid_remove()
                self.vid_pst_rem_val.grid_remove()
                self.vid_pst_ela_val.grid_remove()
                self.vid_pst_ela_lbl.grid_remove()
                self.vid_pst_can_btn.grid_remove()
                self.vid_pst_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.vid_pst_per.grid_configure(pady=(self.pady_progress_window, 0))

        # postprocessing of videos
        elif process == "plt":
            if status == "load":
                self.plt_wai_lbl.configure(text = starting_up_txt[i18n_lang_idx()])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.plt_wai_lbl.grid_remove()
                    self.plt_ela_lbl.grid()
                    self.plt_ela_val.grid()
                    self.plt_rem_lbl.grid()
                    self.plt_rem_val.grid()
                    self.plt_can_btn.grid()
                    self.plt_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.plt_pbr.set(percentage)
                self.plt_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.plt_per.configure(fg_color=(self.green_primary, "#1F6BA5"))
                else:
                    self.plt_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.plt_ela_val.configure(text = time_ela)
                self.plt_rem_val.configure(text = time_rem)
            elif status == "done":
                self.plt_rem_lbl.grid_remove()
                self.plt_rem_val.grid_remove()
                self.plt_ela_val.grid_remove()
                self.plt_ela_lbl.grid_remove()
                self.plt_can_btn.grid_remove()
                self.plt_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.plt_per.grid_configure(pady=(self.pady_progress_window, 0))

        # update screen
        self.progress_top_level_window.update()

    def open(self) -> None:
        self.progress_top_level_window.deiconify()

    def close(self) -> None:
        self.progress_top_level_window.destroy()

