"""AppState — owns all mutable application state previously managed via globals."""
import tkinter as tk


class AppState:
    """Holds all mutable application state previously managed via ``global``.

    Instantiated once after the root ``tk.Tk()`` window is created, since
    tkinter variables require an active Tk instance.  Passed to (or accessed
    by) functions that previously used ``global``.
    """

    def __init__(self):
        # ── Tkinter variables (user-facing settings) ──────────────────
        # Folder selection
        self.var_choose_folder = tk.StringVar()
        self.var_choose_folder_short = tk.StringVar()

        # Detection model
        self.var_det_model = tk.StringVar()
        self.var_det_model_short = tk.StringVar()
        self.var_det_model_path = tk.StringVar()

        # Classification model
        self.var_cls_model = tk.StringVar()

        # Thresholds
        self.var_cls_detec_thresh = tk.DoubleVar(value=0.6)
        self.var_cls_class_thresh = tk.DoubleVar(value=0.6)
        self.var_thresh = tk.DoubleVar(value=0.6)

        # Deploy options
        self.var_use_custom_img_size_for_deploy = tk.BooleanVar(value=False)
        self.var_image_size_for_deploy = tk.StringVar(value="1280")
        self.var_disable_GPU = tk.BooleanVar(value=False)
        self.var_process_img = tk.BooleanVar(value=True)
        self.var_use_checkpnts = tk.BooleanVar(value=False)
        self.var_cont_checkpnt = tk.BooleanVar(value=False)
        self.var_checkpoint_freq = tk.StringVar(value="500")
        self.var_process_vid = tk.BooleanVar(value=False)
        self.var_not_all_frames = tk.BooleanVar(value=False)
        self.var_nth_frame = tk.StringVar(value="10")

        # Postprocessing options
        self.var_separate_files = tk.BooleanVar(value=False)
        self.var_file_placement = tk.IntVar(value=2)
        self.var_sep_conf = tk.BooleanVar(value=False)
        self.var_keep_series = tk.BooleanVar(value=False)
        self.var_vis_files = tk.BooleanVar(value=False)
        self.var_vis_size = tk.StringVar()
        self.var_vis_bbox = tk.BooleanVar(value=True)
        self.var_vis_blur = tk.BooleanVar(value=False)
        self.var_crp_files = tk.BooleanVar(value=False)
        self.var_exp = tk.BooleanVar(value=False)
        self.var_exp_format = tk.StringVar()
        self.var_plt = tk.BooleanVar(value=False)
        self.var_abs_paths = tk.BooleanVar(value=False)

        # Output directory
        self.var_output_dir = tk.StringVar()
        self.var_output_dir_short = tk.StringVar()

        # Classification extras
        self.var_smooth_cls_animal = tk.BooleanVar(value=False)
        self.var_keep_series_seconds = tk.DoubleVar(value=30.0)
        self.var_tax_fallback = tk.BooleanVar(value=True)
        self.var_exclude_subs = tk.BooleanVar(value=False)
        self.var_tax_levels = tk.StringVar()
        self.var_sppnet_location = tk.StringVar()

        # HITL
        self.var_hitl_file_order = tk.IntVar(value=1)

        # ── Non-widget mutable state (previously ``global``) ──────────
        # Cancel / deploy (cancel_var is a plain bool, not a tkinter var)
        self.cancel_var = False
        self.cancel_deploy_model_pressed = False
        self.cancel_speciesnet_deploy_pressed = False
        self.subprocess_output = ""
        self.warn_smooth_vid = False
        self.temp_frame_folder = ""

        # Progress and error tracking
        self.progress_window = None
        self.postprocessing_error_log = []
        self.model_error_log = []
        self.model_warning_log = []
        self.model_special_char_log = []

        # HITL state
        self.selection_dict = {}

        # Dropdown option lists (rebuilt on language change / model refresh)
        self.dpd_options_cls_model = []
        self.dpd_options_model = []
        self.sim_dpd_options_cls_model = []
        self.loc_chkpnt_file = ""

        # Init flags
        self.checkpoint_freq_init = True
        self.image_size_for_deploy_init = True
        self.nth_frame_init = True
        self.shown_abs_paths_warning = True  # starts True; set False after first warning shown
        self.check_mark_one_row = False
        self.check_mark_two_rows = False

        # Timelapse integration
        self.timelapse_mode = False
        self.timelapse_path = ""

        # Caches
        self._all_supported_model_classes_cache = None

        # ── Widget references (set after UI construction) ──────────────
        self.btn_start_deploy = None
        self.sim_run_btn = None
        self.sim_dir_pth = None
        self.sim_mdl_dpd = None
        self.sim_spp_scr = None
        self.rad_ann_var = None          # tk.IntVar, set during HITL window build
        self.hitl_ann_selection_frame = None
        self.hitl_settings_canvas = None
        self.hitl_settings_window = None
        self.lbl_n_total_imgs = None
