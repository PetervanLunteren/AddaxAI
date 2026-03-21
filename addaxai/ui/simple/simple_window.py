"""Simple mode window construction for AddaxAI."""

import webbrowser
import tkinter as tk
from tkinter.font import Font
import tkinter.messagebox as mb
from typing import Any, Callable, Dict, List, Optional

try:
    import customtkinter
    _CTkToplevel = customtkinter.CTkToplevel
    _CTkFrame = customtkinter.CTkFrame
    _CTkFont = customtkinter.CTkFont
    _CTkImage = customtkinter.CTkImage
    _CTkLabel = customtkinter.CTkLabel
    _CTkButton = customtkinter.CTkButton
    _CTkOptionMenu = customtkinter.CTkOptionMenu
except ImportError:
    class _CTkToplevel:
        pass
    class _CTkFrame:
        pass
    class _CTkFont:
        pass
    class _CTkImage:
        pass
    class _CTkLabel:
        pass
    class _CTkButton:
        pass
    class _CTkOptionMenu:
        pass

from addaxai.i18n import t, lang_idx as i18n_lang_idx
from addaxai.ui.widgets.frames import MyMainFrame, MySubFrame, MySubSubFrame
from addaxai.ui.widgets.buttons import InfoButton, GreyTopButton
from addaxai.ui.widgets.species_selection import SpeciesSelectionFrame


def sim_dir_show_info() -> None:
    """Show info dialog for the directory selection panel."""
    mb.showinfo(title=t('information'),
                message=["Select the images to analyse",
                         "Seleccionar las imágenes a analizar",
                         "Sélectionner les images à analyser"][i18n_lang_idx()],
                detail=["Here you can select a folder containing camera trap images. It will process all images it can find, also in subfolders."
                        " Switch to advanced mode for more options.",
                        " Aquí puede seleccionar una carpeta que contenga imágenes de cámaras trampa."
                        " Procesará todas las imágenes que encuentre, también en las subcarpetas. Cambia al modo avanzado para más opciones.",
                        "Ici vous pouvez choisir un dossier contenant les images des pièges photographiques. Toutes les images trouvées, incluant celles "
                        "dans les sous-dossiers seront traitées. Utiliser le mode avancé pour plus d'options."][i18n_lang_idx()])


def sim_spp_show_info() -> None:
    """Show info dialog for the species selection panel."""
    mb.showinfo(title=t('information'),
                message=["Select the species that are present",
                         "Seleccione las especies presentes",
                         "Sélection des espèces présentes"][i18n_lang_idx()],
                detail=["Here, you can select and deselect the animals categories that are present in your project"
                        " area. If the animal category is not selected, it will be excluded from the results. The "
                        "category list will update according to the model selected.",
                        "Aquí puede seleccionar y anular"
                        " la selección de las categorías de animales presentes en la zona de su proyecto. Si la "
                        "categoría de animales no está seleccionada, quedará excluida de los resultados. La lista de "
                        "categorías se actualizará según el modelo seleccionado.",
                        "Ici, vous pouvez sélectionner/désélectionner les catégories d'espèces présentes dans la zone de votre projet."
                        " Si une catégorie n'est pas sélectionnée, elle sera exclue des résultats. La liste de catégories sera mise-à-jour en "
                        "fonction du modèle sélectionné."][i18n_lang_idx()])


def sim_mdl_show_info(var_cls_model: Optional[Any] = None, show_model_info_func: Optional[Callable[[], None]] = None) -> None:
    """Show info dialog for the model selection panel.

    Args:
        var_cls_model: tkinter StringVar holding the current cls model selection
        show_model_info_func: callable to show model info (used when a model is selected)
    """
    cls_val = var_cls_model.get() if var_cls_model is not None else t('none')
    if cls_val == t('none'):
        mb.showinfo(title=t('information'),
                    message=["Select the model to identify animals",
                             "Seleccione el modelo para identificar animales",
                             "Sélectionner le modèle d'identification d'animaux"][i18n_lang_idx()],
                    detail=["Here, you can choose a model that can identify your target species. If you select 'None', it will find vehicles,"
                            " people, and animals, but will not further identify them. When a model is selected, press this button again to "
                            "read more about the model in question.",
                            "Aquí, puede elegir un modelo que pueda identificar su especie objetivo."
                            " Si selecciona 'Ninguno', encontrará vehículos, personas y animales, pero no los identificará más. Cuando haya "
                            "seleccionado un modelo, vuelva a pulsar este botón para obtener más información sobre el modelo en cuestión.",
                            "Ici, vous pouvez choisir un modèle qui identifie les espèces cibles. Si vous sélectionnez 'Aucun', il trouvera les "
                            "véhicules, les personnes et les animaux, mais sans les identifier. Lorsqu'un modèle est sélectionné, cliquer sur ce "
                            "bouton à nouveau pour obtenir plus d'information sur ce dernier."][i18n_lang_idx()])
    else:
        if show_model_info_func is not None:
            show_model_info_func()


def build_simple_mode(root: Any, version: str, addaxai_files: str, scale_factor: float,
                      padx: int, pady: int, yellow_primary: str, green_primary: str,
                      icon_size: int, logo_width: int, logo_height: int,
                      sim_window_width: int, sim_window_height: int, addax_txt_size: int,
                      pil_sidebar: Any, pil_logo_incl_text: Any, pil_dir_image: Any,
                      pil_mdl_image: Any, pil_spp_image: Any, pil_run_image: Any,
                      on_toplevel_close: Callable[[], None], switch_mode: Callable[[], None],
                      set_language: Callable[[], None], sponsor_project: Callable[[], None],
                      reset_values: Callable[[], None], browse_dir_func: Callable[[], None],
                      update_frame_states: Callable[[], None],
                      start_deploy_func: Callable[[], None],
                      sim_mdl_dpd_callback: Callable[[], None],
                      var_choose_folder: Any, var_choose_folder_short: Any,
                      dsp_choose_folder: int, row_choose_folder: int,
                      dpd_options_cls_model: List[str], suffixes_for_sim_none: List[str],
                      global_vars: Dict[str, Any], var_cls_model: Any,
                      show_model_info_func: Callable[[], None],
                      yellow_secondary: str, yellow_tertiary: str,
                      grey_button_border_width: int) -> Dict[str, Any]:
    """Build the simple mode window and all its widgets.

    Returns a dict of all widget references that other code needs.
    """
    # Create CTkImages from PIL images
    dir_image = _CTkImage(pil_dir_image, size=(icon_size, icon_size))
    mdl_image = _CTkImage(pil_mdl_image, size=(icon_size, icon_size))
    spp_image = _CTkImage(pil_spp_image, size=(icon_size, icon_size))
    run_image = _CTkImage(pil_run_image, size=(icon_size, icon_size))

    # set up window
    simple_mode_win = _CTkToplevel(root)
    simple_mode_win.title(f"AddaxAI v{version} - " + t('simple_mode'))
    simple_mode_win.geometry("+20+20")
    simple_mode_win.protocol("WM_DELETE_WINDOW", on_toplevel_close)
    simple_mode_win.columnconfigure(0, weight=1, minsize=500)
    main_label_font = _CTkFont(family='CTkFont', size=14, weight='bold')
    simple_bg_image = _CTkImage(pil_sidebar, size=(sim_window_width, sim_window_height))
    simple_bg_image_label = _CTkLabel(simple_mode_win, text="", image=simple_bg_image)
    simple_bg_image_label.grid(row=0, column=0)
    simple_main_frame = _CTkFrame(simple_mode_win, corner_radius=0, fg_color='transparent')
    simple_main_frame.grid(row=0, column=0, sticky="ns")
    simple_mode_win.withdraw()  # only show when all widgets are loaded

    # logo
    sim_top_banner = _CTkImage(pil_logo_incl_text, size=(logo_width, logo_height))
    _CTkLabel(simple_main_frame, text="", image=sim_top_banner).grid(
        column=0, row=0, columnspan=2, sticky='ew', pady=(pady, 0), padx=0)

    # top buttons
    sim_btn_switch_mode = GreyTopButton(
        master=simple_main_frame, text=t('sim_btn_switch_mode'), command=switch_mode,
        yellow_secondary=yellow_secondary, yellow_tertiary=yellow_tertiary,
        border_width=grey_button_border_width)
    sim_btn_switch_mode.grid(row=0, column=0, padx=padx, pady=(pady, 0), columnspan=2, sticky="nw")
    sim_btn_switch_lang = GreyTopButton(
        master=simple_main_frame, text="Switch language", command=set_language,
        yellow_secondary=yellow_secondary, yellow_tertiary=yellow_tertiary,
        border_width=grey_button_border_width)
    sim_btn_switch_lang.grid(row=0, column=0, padx=padx, pady=(0, 0), columnspan=2, sticky="sw")
    sim_btn_sponsor = GreyTopButton(
        master=simple_main_frame, text=t('adv_btn_sponsor'), command=sponsor_project,
        yellow_secondary=yellow_secondary, yellow_tertiary=yellow_tertiary,
        border_width=grey_button_border_width)
    sim_btn_sponsor.grid(row=0, column=0, padx=padx, pady=(pady, 0), columnspan=2, sticky="ne")
    sim_btn_reset_values = GreyTopButton(
        master=simple_main_frame, text=t('adv_btn_reset_values'), command=reset_values,
        yellow_secondary=yellow_secondary, yellow_tertiary=yellow_tertiary,
        border_width=grey_button_border_width)
    sim_btn_reset_values.grid(row=0, column=0, padx=padx, pady=(0, 0), columnspan=2, sticky="se")

    # choose folder
    sim_dir_frm_1 = MyMainFrame(master=simple_main_frame, scale_factor=scale_factor)
    sim_dir_frm_1.grid(row=2, column=0, padx=padx, pady=pady, sticky="nswe")
    sim_dir_img_widget = _CTkLabel(sim_dir_frm_1, text="", image=dir_image, compound='left')
    sim_dir_img_widget.grid(row=0, column=0, padx=padx, pady=pady, sticky="nswe")
    sim_dir_frm = MySubFrame(master=sim_dir_frm_1)
    sim_dir_frm.grid(row=0, column=1, padx=(0, padx), pady=pady, sticky="nswe")
    sim_dir_lbl = _CTkLabel(sim_dir_frm, text=t('sim_dir_lbl'), font=main_label_font)
    sim_dir_lbl.grid(row=0, column=0, padx=padx, pady=(0, pady / 4), columnspan=2, sticky="nsw")
    sim_dir_inf = InfoButton(master=sim_dir_frm, text="?", command=sim_dir_show_info)
    sim_dir_inf.grid(row=0, column=0, padx=padx, pady=pady, sticky="e", columnspan=2)
    sim_dir_btn = _CTkButton(
        sim_dir_frm, text=t('browse'), width=1,
        command=lambda: [browse_dir_func(var_choose_folder, var_choose_folder_short,
                                         dsp_choose_folder, 25, row_choose_folder, 0, 'w',
                                         source_dir=True),
                         update_frame_states()])
    sim_dir_btn.grid(row=1, column=0, padx=(padx, padx / 2), pady=(0, pady), sticky="nswe")
    sim_dir_pth_frm = MySubSubFrame(master=sim_dir_frm)
    sim_dir_pth_frm.grid(row=1, column=1, padx=(padx / 2, padx), pady=(0, pady), sticky="nesw")
    sim_dir_pth = _CTkLabel(sim_dir_pth_frm, text=t('sim_dir_pth'), text_color="grey")
    sim_dir_pth.pack()

    # choose model
    sim_mdl_frm_1 = MyMainFrame(master=simple_main_frame, scale_factor=scale_factor)
    sim_mdl_frm_1.grid(row=3, column=0, padx=padx, pady=(0, pady), sticky="nswe")
    sim_mdl_img_widget = _CTkLabel(sim_mdl_frm_1, text="", image=mdl_image, compound='left')
    sim_mdl_img_widget.grid(row=1, column=0, padx=padx, pady=pady, sticky="nswe")
    sim_mdl_frm = MySubFrame(master=sim_mdl_frm_1)
    sim_mdl_frm.grid(row=1, column=1, padx=(0, padx), pady=pady, sticky="nswe")
    sim_mdl_lbl = _CTkLabel(sim_mdl_frm, text=t('sim_mdl_lbl'), font=main_label_font)
    sim_mdl_lbl.grid(row=0, column=0, padx=padx, pady=(0, pady / 4), columnspan=2, sticky="nsw")
    sim_mdl_inf = InfoButton(
        master=sim_mdl_frm, text="?",
        command=lambda: sim_mdl_show_info(var_cls_model=var_cls_model,
                                           show_model_info_func=show_model_info_func))
    sim_mdl_inf.grid(row=0, column=0, padx=padx, pady=pady, sticky="e", columnspan=2)
    # convert to more elaborate dpd value for the 'None' simple mode option
    sim_dpd_options_cls_model = [[item[0] + suffixes_for_sim_none[i], *item[1:]]
                                  for i, item in enumerate(dpd_options_cls_model)]
    sim_mdl_dpd = _CTkOptionMenu(
        sim_mdl_frm, values=sim_dpd_options_cls_model[i18n_lang_idx()],
        command=sim_mdl_dpd_callback, width=1)
    sim_mdl_dpd.set(sim_dpd_options_cls_model[i18n_lang_idx()][global_vars["var_cls_model_idx"]])
    sim_mdl_dpd.grid(row=1, column=0, padx=padx, pady=(pady / 4, pady), sticky="nswe", columnspan=2)

    # select animals
    sim_spp_frm_1 = MyMainFrame(master=simple_main_frame, scale_factor=scale_factor)
    sim_spp_frm_1.grid(row=4, column=0, padx=padx, pady=(0, pady), sticky="nswe")
    sim_spp_img_widget = _CTkLabel(sim_spp_frm_1, text="", image=spp_image, compound='left')
    sim_spp_img_widget.grid(row=2, column=0, padx=padx, pady=pady, sticky="nswe")
    sim_spp_frm = MySubFrame(master=sim_spp_frm_1, width=1000)
    sim_spp_frm.grid(row=2, column=1, padx=(0, padx), pady=pady, sticky="nswe")
    sim_spp_lbl = _CTkLabel(sim_spp_frm, text=t('sim_spp_lbl'), font=main_label_font,
                             text_color='grey')
    sim_spp_lbl.grid(row=0, column=0, padx=padx, pady=(0, pady / 4), columnspan=2, sticky="nsw")
    sim_spp_inf = InfoButton(master=sim_spp_frm, text="?", command=sim_spp_show_info)
    sim_spp_inf.grid(row=0, column=0, padx=padx, pady=pady, sticky="e", columnspan=2)
    sim_spp_scr_height = 238
    sim_spp_scr = SpeciesSelectionFrame(master=sim_spp_frm, height=sim_spp_scr_height,
                                         dummy_spp=True, pady=pady)
    sim_spp_scr._scrollbar.configure(height=0)
    sim_spp_scr.grid(row=1, column=0, padx=padx, pady=(pady / 4, pady), sticky="ew",
                     columnspan=2)

    # deploy button
    sim_run_frm_1 = MyMainFrame(master=simple_main_frame, scale_factor=scale_factor)
    sim_run_frm_1.grid(row=5, column=0, padx=padx, pady=(0, pady), sticky="nswe")
    sim_run_img_widget = _CTkLabel(sim_run_frm_1, text="", image=run_image, compound='left')
    sim_run_img_widget.grid(row=3, column=0, padx=padx, pady=pady, sticky="nswe")
    sim_run_frm = MySubFrame(master=sim_run_frm_1, width=1000)
    sim_run_frm.grid(row=3, column=1, padx=(0, padx), pady=pady, sticky="nswe")
    sim_run_btn = _CTkButton(sim_run_frm, text=t('sim_run_btn'),
                              command=lambda: start_deploy_func(simple_mode=True))
    sim_run_btn.grid(row=0, column=0, padx=padx, pady=pady, sticky="nswe", columnspan=2)

    # about
    sim_abo_lbl = tk.Label(simple_main_frame, text=t('adv_abo_lbl'),
                            font=Font(size=addax_txt_size), fg="black", bg=yellow_primary)
    sim_abo_lbl.grid(row=6, column=0, columnspan=2, sticky="")
    sim_abo_lbl_link = tk.Label(simple_main_frame, text="addaxdatascience.com", cursor="hand2",
                                 font=Font(size=addax_txt_size, underline=1),
                                 fg=green_primary, bg=yellow_primary)
    sim_abo_lbl_link.grid(row=7, column=0, columnspan=2, sticky="", pady=(0, pady))
    sim_abo_lbl_link.bind("<Button-1>", lambda e: webbrowser.open_new("http://addaxdatascience.com"))

    return {
        'window': simple_mode_win,
        'btn_switch_mode': sim_btn_switch_mode,
        'btn_switch_lang': sim_btn_switch_lang,
        'btn_sponsor': sim_btn_sponsor,
        'btn_reset_values': sim_btn_reset_values,
        'dir_lbl': sim_dir_lbl,
        'dir_btn': sim_dir_btn,
        'dir_pth': sim_dir_pth,
        'mdl_lbl': sim_mdl_lbl,
        'mdl_dpd': sim_mdl_dpd,
        'mdl_frm': sim_mdl_frm,
        'spp_lbl': sim_spp_lbl,
        'spp_scr': sim_spp_scr,
        'spp_frm': sim_spp_frm,
        'spp_scr_height': sim_spp_scr_height,
        'dpd_options_cls_model': sim_dpd_options_cls_model,
        'run_btn': sim_run_btn,
        'abo_lbl': sim_abo_lbl,
    }
