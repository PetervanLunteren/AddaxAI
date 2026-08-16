"""Tests for the micromamba line-to-progress parser used by setup."""

import pytest

from app.ml.environment_manager import (
    ENV_PROGRESS_FLOOR,
    parse_micromamba_progress,
)

# Realistic progress allocation for env-addaxai-base from setup logs:
# 1 conda package + 15 pip packages -> conda gets 5%-11%, pip 11%-100%.
CONDA_START = 0.05
CONDA_END = 0.11
PIP_START = 0.11
PIP_END = 1.0


def _parse(line: str, current: float = ENV_PROGRESS_FLOOR) -> tuple[float, str]:
    return parse_micromamba_progress(
        line,
        current,
        conda_start=CONDA_START,
        conda_end=CONDA_END,
        pip_start=PIP_START,
        pip_end=PIP_END,
    )


def test_unknown_line_holds_progress_at_floor() -> None:
    """A line that matches no phase pattern must not slide the bar back to 0.

    This is the bug behind the original 'stuck at 0% for several minutes'
    report: every uncategorised libmamba diag line used to reset progress
    to whatever the closure was initialised with (0.0), wiping the
    "Starting package installation" 10% preset.
    """
    progress, caption = _parse(
        "info libmamba Checking for CA certificates at the root prefix",
        current=ENV_PROGRESS_FLOOR,
    )
    assert progress == ENV_PROGRESS_FLOOR
    # Fallback caption is the (truncated) raw line so the user still
    # sees text changing.
    assert "CA certificates" in caption


def test_resolve_phase_crawls_one_to_four_percent() -> None:
    """The resolve phase moves the bar slightly so the user sees life.

    Each phase line lifts progress by ~1%, capped at conda_start.
    """
    p1, c1 = _parse("info libmamba Searching index cache file for repo conda-forge/noarch")
    assert p1 == pytest.approx(ENV_PROGRESS_FLOOR)  # floor wins, since 0.10 > 0.01
    assert c1 == "Loading package index..."

    # From an earlier-stage cold start (no floor yet, e.g. retry path).
    p1c, _ = _parse(
        "info libmamba Searching index cache file for repo conda-forge/noarch",
        current=0.0,
    )
    assert p1c == pytest.approx(0.01)

    p2, c2 = _parse("Fetch Shard Index for conda-forge/noarch", current=0.0)
    assert p2 == pytest.approx(0.02)
    assert c2 == "Loading package index..."

    p3, c3 = _parse("Parsing Packages' Records", current=0.0)
    assert p3 == pytest.approx(0.03)
    assert c3 == "Resolving dependencies..."

    p4, c4 = _parse("Resolving Environment", current=0.0)
    assert p4 == pytest.approx(0.04)
    assert c4 == "Resolving dependencies..."


def test_transaction_lift_to_conda_start() -> None:
    """The 'Transaction' line (without 'starting') moves the bar to conda_start."""
    progress, caption = _parse("Transaction", current=ENV_PROGRESS_FLOOR)
    # max(0.10, 0.05) = 0.10 -- floor still wins for this small env.
    assert progress == pytest.approx(0.10)
    assert caption == "Downloading packages..."

    # But on an env with a much bigger conda slice (so conda_start > 0.10)
    # the transaction lift would actually move the bar past the floor.
    bigger, _ = parse_micromamba_progress(
        "Transaction",
        0.0,
        conda_start=0.30,
        conda_end=0.50,
        pip_start=0.50,
        pip_end=1.0,
    )
    assert bigger == pytest.approx(0.30)


def test_linking_halfway_through_conda() -> None:
    progress, caption = _parse("Linking python-3.11.15", current=ENV_PROGRESS_FLOOR)
    expected = CONDA_START + (CONDA_END - CONDA_START) * 0.5
    assert progress == pytest.approx(max(ENV_PROGRESS_FLOOR, expected))
    assert caption == "Installing packages..."


def test_pip_phase_progression() -> None:
    a, ca = _parse("Installing pip packages...", current=CONDA_END)
    assert a == pytest.approx(PIP_START)
    assert ca == "Installing Python packages..."

    b, cb = _parse("Collecting numpy", current=PIP_START)
    assert b == pytest.approx(PIP_START + (PIP_END - PIP_START) * 0.3)
    assert cb == "Downloading Python packages..."

    c, cc = _parse("Installing collected packages: numpy", current=b)
    assert c == pytest.approx(PIP_START + (PIP_END - PIP_START) * 0.7)
    assert cc == "Installing Python packages..."

    d, cd = _parse("Successfully installed numpy-2.4.4", current=c)
    assert d == pytest.approx(0.95)
    assert cd == "Python packages installed"


def test_pyc_compilation_line_gets_reassuring_caption() -> None:
    """The pyc-compile phase must not freeze the bar on a raw libmamba line.

    This is the beta report: the install went silent at
    'libmamba Waiting for pyc compilation to finish', the bar stayed put,
    and it looked stuck. Now that line lifts the bar toward the end of the
    conda range and shows a caption that says it is finishing.
    """
    progress, caption = _parse(
        "info libmamba Waiting for pyc compilation to finish",
        current=CONDA_START,
    )
    expected = CONDA_START + (CONDA_END - CONDA_START) * 0.85
    assert progress == pytest.approx(max(CONDA_START, expected))
    assert "compiling files" in caption.lower()
    assert "libmamba" not in caption


def test_progress_is_monotonic_under_real_sequence() -> None:
    """Replay the line order from a real setup log and confirm the bar
    only ever moves forward. Any backwards movement is a UX regression.
    """
    # Subset of the real lines observed in env-addaxai-base creation,
    # in chronological order. Interleaved with diag lines to confirm
    # unknown lines don't reset progress.
    script = [
        "info libmamba Searching index cache file for repo 'conda-forge/noarch'",
        "info libmamba Valid cache found for 'conda-forge/noarch': 0",  # unknown
        "Fetch Shard Index for conda-forge/noarch",
        "Fetching and Parsing Packages' Shards",
        "Parsing Packages' Records",
        "info libmamba Loading site packages",  # unknown
        "Resolving Environment",
        "Transaction",
        "Linking python-3.11.15",
        "Linking torch-2.11.0",
        "info libmamba Cleaned 0 .mamba_trash files",  # unknown
        "Installing pip packages...",
        "Collecting numpy",
        "Installing collected packages: numpy",
        "Successfully installed numpy-2.4.4",
    ]
    progress = ENV_PROGRESS_FLOOR
    history: list[float] = [progress]
    for line in script:
        progress, _ = _parse(line, current=progress)
        history.append(progress)
    assert history == sorted(history), f"progress went backwards: {history}"
    # End state must be the explicit 0.95 from 'Successfully installed'.
    assert progress == pytest.approx(0.95)


def test_unknown_line_still_returns_caption() -> None:
    """Long verbose libmamba lines must be truncated, not passed through full length."""
    long_line = "info libmamba " + "x" * 500
    _, caption = _parse(long_line, current=ENV_PROGRESS_FLOOR)
    assert len(caption) <= 80


def test_unknown_line_does_not_lift_progress() -> None:
    """Sanity: an unknown line is supposed to hold progress, not lift it."""
    before = 0.07
    after, _ = _parse("some random libmamba chatter", current=before)
    assert after == before


def test_raw_download_progress_becomes_a_moving_caption() -> None:
    """
    pip prints no progress bar into a pipe, so PIP_PROGRESS_BAR=raw is
    the only sign of life during the 3.4 GB torch download. The bar
    cannot move (these bytes describe one file of many), but the
    caption must, or users conclude setup has frozen and restart it.
    """
    progress, caption = _parse("Progress 1258291200 of 3630000000", current=0.5)

    assert progress == 0.5
    assert caption == "Downloading Python packages (1200 MB of 3461 MB)"


def test_small_downloads_do_not_show_a_zero_megabyte_caption() -> None:
    """Most packages are tiny; "0 MB of 0 MB" would be noise."""
    _, caption = _parse("Progress 4096 of 28000")

    assert caption == "Downloading Python packages..."


def test_malformed_progress_line_falls_back_to_the_raw_text() -> None:
    _, caption = _parse("Progress lots of bytes")

    assert caption == "Progress lots of bytes"
