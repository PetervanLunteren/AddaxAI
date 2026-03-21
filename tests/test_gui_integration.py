r"""
GUI integration tests: language cycling, mode switching, folder selection.

These tests launch the full GUI via gui_test_runner.py, which schedules
programmatic interactions via root.after() and writes results to a JSON file.

Requires the installed env-base Python -- skipped automatically if not found.
Run with:
    C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe -m pytest tests/test_gui_integration.py -v
Or from .venv (tests will be skipped if env-base not found):
    .venv/Scripts/python -m pytest tests/test_gui_integration.py -v
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

ENV_BASE_PYTHON = r"C:\Users\Topam\AddaxAI_files\envs\env-base\python.exe"
TEST_RUNNER = os.path.join(os.path.dirname(__file__), "gui_test_runner.py")
TIMEOUT_SECONDS = 60  # generous timeout for GUI boot + test execution


@pytest.fixture
def env_base_python():
    if not os.path.isfile(ENV_BASE_PYTHON):
        pytest.skip(f"env-base Python not found at {ENV_BASE_PYTHON}")
    return ENV_BASE_PYTHON


def _run_gui_test(env_python, test_name):
    """Launch gui_test_runner.py with the given test name, return results dict."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix=f"addaxai_test_{test_name}_"
    ) as tmp:
        results_file = tmp.name

    try:
        proc = subprocess.run(
            [env_python, os.path.abspath(TEST_RUNNER), test_name, results_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SECONDS,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        if not os.path.isfile(results_file) or os.path.getsize(results_file) == 0:
            stderr = proc.stderr.decode(errors="replace")
            pytest.fail(
                f"Test runner did not produce results.\n"
                f"Exit code: {proc.returncode}\n"
                f"--- stderr ---\n{stderr}"
            )

        with open(results_file, "r") as f:
            results = json.load(f)

        return results, proc

    except subprocess.TimeoutExpired:
        pytest.fail(
            f"GUI test '{test_name}' timed out after {TIMEOUT_SECONDS}s. "
            f"The GUI may have hung or mainloop was not terminated."
        )

    finally:
        try:
            os.unlink(results_file)
        except OSError:
            pass


def _format_failures(results):
    """Format failed checks for assertion messages."""
    lines = []
    if results.get("error"):
        lines.append(f"Error: {results['error']}")
    for check in results.get("checks", []):
        if check.get("status") == "FAIL":
            lines.append(
                f"  FAIL [{check.get('lang', check.get('step', '?'))}] "
                f"{check.get('widget', check.get('detail', ''))}: "
                f"expected={check.get('expected', '?')!r}, "
                f"actual={check.get('actual', '?')!r}"
            )
    return "\n".join(lines)


# ─── Tests ────────────────────────────────────────────────────────────


def test_language_cycling(env_base_python):
    """Cycle EN→ES→FR→EN and verify key widget texts update correctly."""
    results, proc = _run_gui_test(env_base_python, "language_cycling")

    if not results["passed"]:
        pytest.fail(
            f"Language cycling test failed:\n{_format_failures(results)}"
        )

    # Verify we actually checked widgets (not just a vacuous pass)
    actual_checks = [c for c in results["checks"] if c["status"] == "pass"]
    assert len(actual_checks) >= 10, (
        f"Expected at least 10 passing checks, got {len(actual_checks)}. "
        f"Some widgets may not have been found."
    )


def test_mode_switching(env_base_python):
    """Switch advanced↔simple mode and verify window visibility toggles."""
    results, proc = _run_gui_test(env_base_python, "mode_switching")

    if not results["passed"]:
        pytest.fail(
            f"Mode switching test failed:\n{_format_failures(results)}"
        )

    # Should have at least 4 checks (initial, first switch, visibility, second switch)
    assert len(results["checks"]) >= 4, (
        f"Expected at least 4 checks, got {len(results['checks'])}"
    )


def test_folder_selection(env_base_python):
    """Set source folder and verify frame states update without crashing."""
    results, proc = _run_gui_test(env_base_python, "folder_selection")

    if not results["passed"]:
        pytest.fail(
            f"Folder selection test failed:\n{_format_failures(results)}"
        )


def test_model_dropdown_population(env_base_python):
    """Call update_model_dropdowns() and verify state.dpd_options_* are populated."""
    results, proc = _run_gui_test(env_base_python, "model_dropdown_population")

    if not results["passed"]:
        pytest.fail(
            f"Model dropdown population test failed:\n{_format_failures(results)}"
        )

    checks_passed = [c for c in results["checks"] if c["status"] == "pass"]
    assert len(checks_passed) == 3, (
        f"Expected 3 passing checks (model, cls, sim), got {len(checks_passed)}"
    )


def test_toggle_frames(env_base_python):
    """Toggle sep/vis postprocessing frames on/off and verify no crash."""
    results, proc = _run_gui_test(env_base_python, "toggle_frames")

    if not results["passed"]:
        pytest.fail(
            f"Toggle frames test failed:\n{_format_failures(results)}"
        )

    assert len(results["checks"]) >= 4, (
        f"Expected at least 4 toggle checks, got {len(results['checks'])}"
    )


def test_reset_values(env_base_python):
    """Set vars to non-defaults, call reset_values(), verify they revert."""
    results, proc = _run_gui_test(env_base_python, "reset_values")

    if not results["passed"]:
        pytest.fail(
            f"Reset values test failed:\n{_format_failures(results)}"
        )

    checks_passed = [c for c in results["checks"] if c["status"] == "pass"]
    assert len(checks_passed) == 5, (
        f"Expected 5 passing reset checks, got {len(checks_passed)}"
    )


def test_deploy_validation(env_base_python):
    """Verify start_deploy() shows an error (not crash) when folder has no images."""
    results, proc = _run_gui_test(env_base_python, "deploy_validation")

    if not results["passed"]:
        pytest.fail(
            f"Deploy validation test failed:\n{_format_failures(results)}"
        )


def test_state_attributes(env_base_python):
    """Verify AppState attributes are accessible and correctly initialized after GUI boot."""
    results, proc = _run_gui_test(env_base_python, "state_attributes")

    if not results["passed"]:
        pytest.fail(
            f"State attributes test failed:\n{_format_failures(results)}"
        )

    checks_passed = [c for c in results["checks"] if c["status"] == "pass"]
    assert len(checks_passed) >= 22, (
        f"Expected at least 22 passing attribute checks, got {len(checks_passed)}"
    )
