"""AppState — owns all mutable application state previously managed via globals."""
import tkinter as tk
from typing import Any, Dict, List, Optional


class AppState:
    """Holds all mutable application state previously managed via ``global``.

    Instantiated once after the root ``tk.Tk()`` window is created, since
    tkinter variables require an active Tk instance.  Passed to (or accessed
    by) functions that previously used ``global``.
    """

    def __init__(self) -> None:
        # ── Tkinter variables (user-facing settings) ──────────────────
        # Folder selection
        self.var_choose_folder: tk.StringVar = tk.StringVar()
        self.var_choose_folder_short: tk.StringVar = tk.StringVar()

        # Detection model
        self.var_det_model: tk.StringVar = tk.StringVar()
        self.var_det_model_short: tk.StringVar = tk.StringVar()
        self.var_det_model_path: tk.StringVar = tk.StringVar()

        # Classification model
        self.var_cls_model: tk.StringVar = tk.StringVar()

        # Thresholds
        self.var_cls_detec_thresh: tk.DoubleVar = tk.DoubleVar(value=0.6)
        self.var_cls_class_thresh: tk.DoubleVar = tk.DoubleVar(value=0.6)
        self.var_thresh: tk.DoubleVar = tk.DoubleVar(value=0.6)

        # Deploy options
        self.var_use_custom_img_size_for_deploy: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_image_size_for_deploy: tk.StringVar = tk.StringVar(value="1280")
        self.var_disable_GPU: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_process_img: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_use_checkpnts: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_cont_checkpnt: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_checkpoint_freq: tk.StringVar = tk.StringVar(value="500")
        self.var_process_vid: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_not_all_frames: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_nth_frame: tk.StringVar = tk.StringVar(value="10")

        # Postprocessing options
        self.var_separate_files: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_file_placement: tk.IntVar = tk.IntVar(value=2)
        self.var_sep_conf: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_keep_series: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_vis_files: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_vis_size: tk.StringVar = tk.StringVar()
        self.var_vis_bbox: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_vis_blur: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_crp_files: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_exp: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_exp_format: tk.StringVar = tk.StringVar()
        self.var_plt: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_abs_paths: tk.BooleanVar = tk.BooleanVar(value=False)

        # Output directory
        self.var_output_dir: tk.StringVar = tk.StringVar()
        self.var_output_dir_short: tk.StringVar = tk.StringVar()

        # Classification extras
        self.var_smooth_cls_animal: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_keep_series_seconds: tk.DoubleVar = tk.DoubleVar(value=30.0)
        self.var_tax_fallback: tk.BooleanVar = tk.BooleanVar(value=True)
        self.var_exclude_subs: tk.BooleanVar = tk.BooleanVar(value=False)
        self.var_tax_levels: tk.StringVar = tk.StringVar()
        self.var_sppnet_location: tk.StringVar = tk.StringVar()

        # HITL
        self.var_hitl_file_order: tk.IntVar = tk.IntVar(value=1)

        # ── Non-widget mutable state (previously ``global``) ──────────
        # Cancel / deploy (cancel_var is a plain bool, not a tkinter var)
        self.cancel_var: bool = False
        self.cancel_deploy_model_pressed: bool = False
        self.cancel_speciesnet_deploy_pressed: bool = False
        self.subprocess_output: str = ""
        self.warn_smooth_vid: bool = True   # reset to True each app start; set False after shown
        self.temp_frame_folder: str = ""

        # Progress and error tracking
        self.progress_window: Optional[Any] = None
        self.postprocessing_error_log: str = ""  # file path string, not a list
        self.model_error_log: str = ""       # file path string
        self.model_warning_log: str = ""     # file path string
        self.model_special_char_log: str = ""  # file path string

        # HITL state
        self.selection_dict: Dict[str, Any] = {}

        # Dropdown option lists (rebuilt on language change / model refresh)
        self.dpd_options_cls_model: List[str] = []
        self.dpd_options_model: List[str] = []
        self.sim_dpd_options_cls_model: List[str] = []
        self.loc_chkpnt_file: str = ""

        # Init flags
        self.checkpoint_freq_init: bool = True
        self.image_size_for_deploy_init: bool = True
        self.nth_frame_init: bool = True
        self.shown_abs_paths_warning: bool = True  # starts True; set False after first warning shown
        self.check_mark_one_row: bool = False
        self.check_mark_two_rows: bool = False

        # Timelapse integration
        self.timelapse_mode: bool = False
        self.timelapse_path: str = ""

        # Caches
        self._all_supported_model_classes_cache: Optional[Any] = None

        # ── Widget references (set after UI construction) ──────────────
        self.btn_start_deploy: Optional[Any] = None
        self.sim_run_btn: Optional[Any] = None
        self.sim_dir_pth: Optional[Any] = None
        self.sim_mdl_dpd: Optional[Any] = None
        self.sim_spp_scr: Optional[Any] = None
        self.rad_ann_var: Optional[tk.IntVar] = None          # tk.IntVar, set during HITL window build
        self.hitl_ann_selection_frame: Optional[Any] = None
        self.hitl_settings_canvas: Optional[Any] = None
        self.hitl_settings_window: Optional[Any] = None
        self.lbl_n_total_imgs: Optional[Any] = None
