"""Tests for addaxai.ui.simple.simple_window — build_simple_mode and sim info functions."""

import inspect
import pytest


def test_import_build_simple_mode():
    from addaxai.ui.simple.simple_window import build_simple_mode


def test_import_sim_dir_show_info():
    from addaxai.ui.simple.simple_window import sim_dir_show_info


def test_import_sim_spp_show_info():
    from addaxai.ui.simple.simple_window import sim_spp_show_info


def test_import_sim_mdl_show_info():
    from addaxai.ui.simple.simple_window import sim_mdl_show_info


def test_sim_dir_show_info_is_callable():
    from addaxai.ui.simple.simple_window import sim_dir_show_info
    assert callable(sim_dir_show_info)


def test_sim_spp_show_info_is_callable():
    from addaxai.ui.simple.simple_window import sim_spp_show_info
    assert callable(sim_spp_show_info)


def test_sim_mdl_show_info_is_callable():
    from addaxai.ui.simple.simple_window import sim_mdl_show_info
    assert callable(sim_mdl_show_info)


def test_build_simple_mode_accepts_root():
    from addaxai.ui.simple.simple_window import build_simple_mode
    sig = inspect.signature(build_simple_mode)
    assert "root" in sig.parameters


def test_build_simple_mode_accepts_scale_factor():
    from addaxai.ui.simple.simple_window import build_simple_mode
    sig = inspect.signature(build_simple_mode)
    assert "scale_factor" in sig.parameters


def test_build_simple_mode_accepts_callbacks():
    from addaxai.ui.simple.simple_window import build_simple_mode
    sig = inspect.signature(build_simple_mode)
    assert "on_toplevel_close" in sig.parameters
    assert "switch_mode" in sig.parameters
    assert "start_deploy_func" in sig.parameters
    assert "sim_mdl_dpd_callback" in sig.parameters


def test_build_simple_mode_accepts_pil_images():
    from addaxai.ui.simple.simple_window import build_simple_mode
    sig = inspect.signature(build_simple_mode)
    assert "pil_dir_image" in sig.parameters
    assert "pil_mdl_image" in sig.parameters
    assert "pil_spp_image" in sig.parameters
    assert "pil_run_image" in sig.parameters
