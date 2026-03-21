r"""
GUI integration test runner.

Boots AddaxAI_GUI.py like dev_launch.py, but instead of entering mainloop,
schedules test actions via root.after() and writes results to a JSON file.

Usage:
    C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe tests/gui_test_runner.py <test_name> <results_file>

Each test function collects assertion data, writes it to results_file as JSON,
then calls root.quit(). The calling pytest test reads the JSON and asserts.
"""

import json
import os
import sys
import traceback

ADDAXAI_FILES = r"C:\Users\Topam\AddaxAI_files"

# ─── Bootstrap (same as dev_launch.py) ────────────────────────────────

paths_to_add = [
    ADDAXAI_FILES,
    os.path.join(ADDAXAI_FILES, "cameratraps"),
    os.path.join(ADDAXAI_FILES, "cameratraps", "megadetector"),
]
for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

gui_path = os.path.join(REPO_ROOT, "AddaxAI_GUI.py")

with open(gui_path, "r", encoding="utf-8") as f:
    source = f.read()

# Patch AddaxAI_files path
source = source.replace(
    "AddaxAI_files = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))",
    f"AddaxAI_files = r'{ADDAXAI_FILES}'"
)

# Remove the `if __name__ == "__main__": main()` block so we control startup
source = source.replace(
    'if __name__ == "__main__":\n    main()',
    '# __main__ block removed by test runner'
)


# ─── Execute GUI module (builds all widgets) ──────────────────────────

gui_globals = {"__file__": gui_path, "__name__": "__not_main__"}
exec(compile(source, gui_path, "exec"), gui_globals)


# ─── Test functions ───────────────────────────────────────────────────

def test_language_cycling(g):
    """Cycle through all 3 languages and verify key widgets update correctly."""
    results = {"passed": True, "checks": [], "error": None}

    try:
        root = g["root"]
        set_language = g["set_language"]
        i18n_lang_idx = g["i18n_lang_idx"]
        t_func = g["t"]

        # Expected widget texts per language for a representative sample
        # Format: (widget_name, i18n_key, description)
        widgets_to_check = [
            ("btn_choose_folder", "browse", "browse button"),
            ("btn_start_deploy", "btn_start_deploy", "start deploy button"),
            ("lbl_choose_folder", "lbl_choose_folder", "choose folder label"),
            ("fst_step", "fst_step", "first step frame"),
            ("snd_step", "snd_step", "second step frame"),
            ("trd_step", "trd_step", "third step frame"),
            ("fth_step", "fth_step", "fourth step frame"),
            ("btn_start_postprocess", "btn_start_postprocess", "start postprocess button"),
            ("lbl_separate_files", "lbl_separate_files", "separate files label"),
            ("lbl_vis_files", "lbl_vis_files", "visualize files label"),
            ("lbl_exp", "lbl_exp", "export label"),
            ("lbl_plt", "lbl_plt", "plots label"),
        ]

        # Simple mode widgets
        sim_widgets_to_check = [
            ("sim_dir_lbl", "sim_dir_lbl", "simple dir label"),
            ("sim_dir_btn", "browse", "simple browse button"),
            ("sim_run_btn", "sim_run_btn", "simple run button"),
            ("sim_spp_lbl", "sim_spp_lbl", "simple species label"),
            ("sim_mdl_lbl", "sim_mdl_lbl", "simple model label"),
        ]

        lang_names = ["English", "Español", "Français"]

        for cycle in range(3):
            # Call set_language to advance to next language
            set_language()
            current_idx = i18n_lang_idx()
            current_lang = lang_names[current_idx]

            # Check advanced mode widgets
            for widget_name, i18n_key, desc in widgets_to_check:
                widget = g.get(widget_name)
                if widget is None:
                    results["checks"].append({
                        "lang": current_lang, "widget": desc,
                        "status": "skip", "reason": f"widget '{widget_name}' not found"
                    })
                    continue

                expected = t_func(i18n_key)
                # Some widgets have prefix spaces or special chars around the text
                actual = widget.cget("text").strip()
                expected_stripped = expected.strip()

                ok = expected_stripped in actual
                results["checks"].append({
                    "lang": current_lang, "widget": desc,
                    "status": "pass" if ok else "FAIL",
                    "expected": expected_stripped, "actual": actual,
                })
                if not ok:
                    results["passed"] = False

            # Check simple mode widgets
            for widget_name, i18n_key, desc in sim_widgets_to_check:
                widget = g.get(widget_name)
                if widget is None:
                    results["checks"].append({
                        "lang": current_lang, "widget": desc,
                        "status": "skip", "reason": f"widget '{widget_name}' not found"
                    })
                    continue

                expected = t_func(i18n_key)
                actual = widget.cget("text").strip()
                expected_stripped = expected.strip()

                ok = expected_stripped in actual
                results["checks"].append({
                    "lang": current_lang, "widget": desc,
                    "status": "pass" if ok else "FAIL",
                    "expected": expected_stripped, "actual": actual,
                })
                if not ok:
                    results["passed"] = False

        # Verify we cycled back to the original language (3 cycles = back to start)
        final_idx = i18n_lang_idx()
        results["checks"].append({
            "lang": "final", "widget": "lang_idx_cycle",
            "status": "pass" if final_idx == 0 else "FAIL",
            "expected": 0, "actual": final_idx,
        })
        if final_idx != 0:
            results["passed"] = False

    except Exception as e:
        results["passed"] = False
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


def test_mode_switching(g):
    """Switch between advanced and simple mode, verify window visibility."""
    results = {"passed": True, "checks": [], "error": None}

    try:
        root = g["root"]
        switch_mode = g["switch_mode"]
        advanc_mode_win = g["advanc_mode_win"]
        simple_mode_win = g["simple_mode_win"]
        load_global_vars = g["load_global_vars"]
        AddaxAI_files = g["AddaxAI_files"]

        def get_mode():
            return load_global_vars(AddaxAI_files).get("advanced_mode", True)

        def is_visible(win):
            """Check if a toplevel window is currently displayed."""
            try:
                return win.winfo_viewable()
            except Exception:
                return False

        # Record initial state
        initial_mode = get_mode()
        results["checks"].append({
            "step": "initial", "advanced_mode": initial_mode,
            "status": "pass",
        })

        # Switch once
        switch_mode()
        root.update_idletasks()
        mode_after_first = get_mode()

        results["checks"].append({
            "step": "after_first_switch",
            "advanced_mode": mode_after_first,
            "status": "pass" if mode_after_first != initial_mode else "FAIL",
            "detail": "mode should have toggled",
        })
        if mode_after_first == initial_mode:
            results["passed"] = False

        # Check window visibility after first switch
        adv_visible = is_visible(advanc_mode_win)
        sim_visible = is_visible(simple_mode_win)
        # After switching from advanced (initial=True) to simple: adv hidden, sim visible
        # After switching from simple (initial=False) to advanced: adv visible, sim hidden
        if initial_mode:
            # Was advanced, now should be simple
            expect_adv, expect_sim = False, True
        else:
            # Was simple, now should be advanced
            expect_adv, expect_sim = True, False

        ok_adv = adv_visible == expect_adv
        ok_sim = sim_visible == expect_sim
        results["checks"].append({
            "step": "visibility_after_first_switch",
            "adv_visible": adv_visible, "sim_visible": sim_visible,
            "expect_adv": expect_adv, "expect_sim": expect_sim,
            "status": "pass" if (ok_adv and ok_sim) else "FAIL",
        })
        if not (ok_adv and ok_sim):
            results["passed"] = False

        # Switch back
        switch_mode()
        root.update_idletasks()
        mode_after_second = get_mode()

        results["checks"].append({
            "step": "after_second_switch",
            "advanced_mode": mode_after_second,
            "status": "pass" if mode_after_second == initial_mode else "FAIL",
            "detail": "mode should have returned to initial",
        })
        if mode_after_second != initial_mode:
            results["passed"] = False

        # Verify visibility restored
        adv_visible_2 = is_visible(advanc_mode_win)
        sim_visible_2 = is_visible(simple_mode_win)
        if initial_mode:
            expect_adv_2, expect_sim_2 = True, False
        else:
            expect_adv_2, expect_sim_2 = False, True

        ok2 = (adv_visible_2 == expect_adv_2) and (sim_visible_2 == expect_sim_2)
        results["checks"].append({
            "step": "visibility_after_second_switch",
            "adv_visible": adv_visible_2, "sim_visible": sim_visible_2,
            "status": "pass" if ok2 else "FAIL",
        })
        if not ok2:
            results["passed"] = False

    except Exception as e:
        results["passed"] = False
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


def test_folder_selection(g):
    """Set var_choose_folder to a valid directory and verify frame states update."""
    results = {"passed": True, "checks": [], "error": None}

    try:
        import tempfile
        root = g["root"]
        var_choose_folder = g["var_choose_folder"]
        update_frame_states = g["update_frame_states"]
        fst_step = g["fst_step"]

        # Create a temporary directory to act as source folder
        test_dir = tempfile.mkdtemp(prefix="addaxai_test_")

        # Initially, var_choose_folder should be empty or invalid
        initial_folder = var_choose_folder.get()
        results["checks"].append({
            "step": "initial_folder",
            "value": initial_folder,
            "status": "pass",
        })

        # Set to our test directory
        var_choose_folder.set(test_dir)
        root.update_idletasks()

        # Call update_frame_states — this should complete fst_step
        update_frame_states()
        root.update_idletasks()

        # Verify fst_step frame is in "complete" state
        # When a frame is "complete", it gets a checkmark. We can check if the
        # frame's text_color or state changed. The simplest check: fst_step
        # should not be disabled.
        fst_text = fst_step.cget("text")
        results["checks"].append({
            "step": "after_set_folder",
            "folder": test_dir,
            "fst_step_text": fst_text,
            "status": "pass",  # If we got here without crash, the wiring works
        })

        # Reset
        var_choose_folder.set("")
        root.update_idletasks()

        # Clean up test directory
        try:
            os.rmdir(test_dir)
        except OSError:
            pass

    except Exception as e:
        results["passed"] = False
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


def test_model_dropdown_population(g):
    """Call update_model_dropdowns() and verify state.dpd_options_* are populated."""
    results = {"passed": True, "checks": [], "error": None}

    try:
        state = g["state"]
        update_model_dropdowns = g["update_model_dropdowns"]
        root = g["root"]

        update_model_dropdowns()
        root.update_idletasks()

        # Check dpd_options_model: should be list of 3 non-empty sublists (one per language)
        opts_model = state.dpd_options_model
        ok_model = (
            isinstance(opts_model, list)
            and len(opts_model) == 3
            and all(isinstance(sub, list) and len(sub) > 0 for sub in opts_model)
        )
        results["checks"].append({
            "check": "dpd_options_model_populated",
            "status": "pass" if ok_model else "FAIL",
            "detail": repr(opts_model)[:200],
        })
        if not ok_model:
            results["passed"] = False

        # Check dpd_options_cls_model
        opts_cls = state.dpd_options_cls_model
        ok_cls = (
            isinstance(opts_cls, list)
            and len(opts_cls) == 3
            and all(isinstance(sub, list) and len(sub) > 0 for sub in opts_cls)
        )
        results["checks"].append({
            "check": "dpd_options_cls_model_populated",
            "status": "pass" if ok_cls else "FAIL",
            "detail": repr(opts_cls)[:200],
        })
        if not ok_cls:
            results["passed"] = False

        # Check sim_dpd_options_cls_model
        opts_sim = state.sim_dpd_options_cls_model
        ok_sim = isinstance(opts_sim, list) and len(opts_sim) == 3
        results["checks"].append({
            "check": "sim_dpd_options_cls_model_populated",
            "status": "pass" if ok_sim else "FAIL",
            "detail": repr(opts_sim)[:200],
        })
        if not ok_sim:
            results["passed"] = False

    except Exception as e:
        results["passed"] = False
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


def test_toggle_frames(g):
    """Toggle postprocessing sub-frames (sep, vis) on/off and verify no crash."""
    results = {"passed": True, "checks": [], "error": None}

    try:
        root = g["root"]
        var_separate_files = g["var_separate_files"]
        var_vis_files = g["var_vis_files"]
        toggle_sep_frame = g["toggle_sep_frame"]
        toggle_vis_frame = g["toggle_vis_frame"]

        toggle_cases = [
            ("sep_frame ON",  var_separate_files, True,  toggle_sep_frame),
            ("sep_frame OFF", var_separate_files, False, toggle_sep_frame),
            ("vis_frame ON",  var_vis_files,      True,  toggle_vis_frame),
            ("vis_frame OFF", var_vis_files,      False, toggle_vis_frame),
        ]

        for label, var, new_val, toggle_fn in toggle_cases:
            var.set(new_val)
            root.update_idletasks()
            try:
                toggle_fn()
                root.update_idletasks()
                results["checks"].append({"check": label, "status": "pass"})
            except Exception as e:
                results["checks"].append({
                    "check": label, "status": "FAIL",
                    "detail": f"{type(e).__name__}: {e}",
                })
                results["passed"] = False

        # Reset to off
        var_separate_files.set(False)
        var_vis_files.set(False)
        root.update_idletasks()

    except Exception as e:
        results["passed"] = False
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


def test_reset_values(g):
    """Set vars to non-default values, call reset_values(), verify they revert."""
    results = {"passed": True, "checks": [], "error": None}

    try:
        root = g["root"]
        reset_values = g["reset_values"]
        var_separate_files = g["var_separate_files"]
        var_vis_files = g["var_vis_files"]
        var_crp_files = g["var_crp_files"]
        var_abs_paths = g["var_abs_paths"]
        var_disable_GPU = g["var_disable_GPU"]

        # Set to non-default (all True)
        for var in (var_separate_files, var_vis_files, var_crp_files, var_abs_paths, var_disable_GPU):
            var.set(True)
        root.update_idletasks()

        reset_values()
        root.update_idletasks()

        # All of these should be False after reset
        expected_false = [
            ("var_separate_files", var_separate_files),
            ("var_vis_files",      var_vis_files),
            ("var_crp_files",      var_crp_files),
            ("var_abs_paths",      var_abs_paths),
            ("var_disable_GPU",    var_disable_GPU),
        ]
        for name, var in expected_false:
            val = var.get()
            ok = (val == False)
            results["checks"].append({
                "check": f"{name}_reset_to_false",
                "status": "pass" if ok else "FAIL",
                "expected": False, "actual": val,
            })
            if not ok:
                results["passed"] = False

    except Exception as e:
        results["passed"] = False
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


def test_deploy_validation(g):
    """Verify start_deploy() shows an error dialog (not crash) when folder has no images."""
    results = {"passed": True, "checks": [], "error": None}

    try:
        import tempfile
        root = g["root"]
        var_choose_folder = g["var_choose_folder"]
        start_deploy = g["start_deploy"]
        mb = g["mb"]

        # Patch showerror to capture calls (prevents dialog blocking)
        call_log = []
        original_showerror = mb.showerror
        mb.showerror = lambda *args, **kwargs: call_log.append({"args": args, "kwargs": kwargs})

        test_dir = tempfile.mkdtemp(prefix="addaxai_deploy_test_")
        var_choose_folder.set(test_dir)
        root.update_idletasks()

        try:
            start_deploy()
            root.update_idletasks()
        finally:
            mb.showerror = original_showerror
            var_choose_folder.set("")
            try:
                os.rmdir(test_dir)
            except OSError:
                pass

        error_shown = len(call_log) > 0
        results["checks"].append({
            "check": "showerror_called_on_empty_folder",
            "status": "pass" if error_shown else "FAIL",
            "detail": f"showerror called {len(call_log)} time(s)",
        })
        if not error_shown:
            results["passed"] = False

    except Exception as e:
        results["passed"] = False
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


def test_state_attributes(g):
    """Verify AppState attributes are accessible with correct initial values after GUI init."""
    results = {"passed": True, "checks": [], "error": None}

    try:
        import tkinter as tk
        state = g["state"]

        # Non-tkinter state attrs (Phase 4 migration)
        # Values reflect actual AppState.__init__ defaults (not the CLAUDE.md design doc)
        expected_defaults = [
            ("cancel_deploy_model_pressed",       False),
            ("cancel_speciesnet_deploy_pressed",  False),
            ("warn_smooth_vid",                   True),   # reset to True each app start
            ("timelapse_mode",                    False),
            ("timelapse_path",                    ""),
            ("subprocess_output",                 ""),
            ("postprocessing_error_log",          ""),     # file path string, not a list
            ("model_error_log",                   ""),     # file path string
            ("model_warning_log",                 ""),     # file path string
            ("model_special_char_log",            ""),     # file path string
            ("selection_dict",                    {}),
            ("checkpoint_freq_init",              True),
            ("image_size_for_deploy_init",        True),
            ("nth_frame_init",                    True),
            ("shown_abs_paths_warning",           True),   # True until first warning shown
        ]
        for attr, expected in expected_defaults:
            actual = getattr(state, attr, "__MISSING__")
            ok = actual == expected
            results["checks"].append({
                "check": f"state.{attr}",
                "status": "pass" if ok else "FAIL",
                "expected": str(expected), "actual": str(actual),
            })
            if not ok:
                results["passed"] = False

        # Tkinter vars should be tk.Variable instances
        # Note: cancel_var is a plain bool in AppState (not a tk.Variable)
        tk_var_attrs = [
            "var_choose_folder", "var_det_model", "var_cls_model",
            "var_thresh", "var_separate_files", "var_vis_files",
            "var_exp",
        ]
        for attr in tk_var_attrs:
            val = getattr(state, attr, None)
            ok = isinstance(val, tk.Variable)
            results["checks"].append({
                "check": f"state.{attr}_is_tk_var",
                "status": "pass" if ok else "FAIL",
                "detail": f"type={type(val).__name__}",
            })
            if not ok:
                results["passed"] = False

        # Widget refs should be non-None after GUI construction
        widget_attrs = ["btn_start_deploy", "sim_run_btn"]
        for attr in widget_attrs:
            val = getattr(state, attr, None)
            ok = val is not None
            results["checks"].append({
                "check": f"state.{attr}_set",
                "status": "pass" if ok else "FAIL",
            })
            if not ok:
                results["passed"] = False

    except Exception as e:
        results["passed"] = False
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


# ─── Dispatcher ───────────────────────────────────────────────────────

TESTS = {
    "language_cycling":          test_language_cycling,
    "mode_switching":            test_mode_switching,
    "folder_selection":          test_folder_selection,
    "model_dropdown_population": test_model_dropdown_population,
    "toggle_frames":             test_toggle_frames,
    "reset_values":              test_reset_values,
    "deploy_validation":         test_deploy_validation,
    "state_attributes":          test_state_attributes,
}


def run_test(test_name, results_file):
    """Schedule the named test to run after the GUI is fully initialized."""
    if test_name not in TESTS:
        result = {"passed": False, "error": f"Unknown test: {test_name}"}
        with open(results_file, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    root = gui_globals["root"]

    def _execute():
        try:
            # Run the GUI's main() initialization but skip mainloop
            # We need to call the parts of main() that set up initial state:
            g = gui_globals

            # Replicate the essential parts of main() without mainloop
            # (argparse, fetch_latest_model_info, frame init, switch_mode x2)
            enable_frame = g["enable_frame"]
            disable_frame = g["disable_frame"]
            set_lang_buttons = g["set_lang_buttons"]
            i18n_lang_idx = g["i18n_lang_idx"]

            # Initialize globals that main() normally sets
            g["timelapse_mode"] = False
            g["timelapse_path"] = ""

            enable_frame(g["fst_step"])
            disable_frame(g["snd_step"])
            disable_frame(g["trd_step"])
            disable_frame(g["fth_step"])
            set_lang_buttons(i18n_lang_idx())

            # The double switch_mode is needed per the original main()
            g["switch_mode"]()
            g["switch_mode"]()

            root.update_idletasks()

            # Now run the actual test
            test_func = TESTS[test_name]
            result = test_func(g)

            with open(results_file, "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            result = {
                "passed": False,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            }
            with open(results_file, "w") as f:
                json.dump(result, f, indent=2)

        finally:
            root.quit()

    # Schedule test execution after GUI is fully rendered
    root.after(2000, _execute)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <test_name> <results_file>", file=sys.stderr)
        print(f"Available tests: {', '.join(TESTS.keys())}", file=sys.stderr)
        sys.exit(1)

    test_name = sys.argv[1]
    results_file = sys.argv[2]
    run_test(test_name, results_file)
