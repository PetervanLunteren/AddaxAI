"""Tests for addaxai.analysis.plots — plotting and visualization utilities."""

import os
import pytest
import pandas as pd
from PIL import Image

from addaxai.analysis.plots import (
    calculate_time_span,
    overlay_logo,
)


# --- calculate_time_span ---

def test_calculate_time_span_basic():
    df = pd.DataFrame({
        "DateTimeOriginal": pd.to_datetime([
            "2022-01-01", "2022-06-15", "2023-01-01",
        ]),
    })
    years, months, weeks, days = calculate_time_span(df)
    assert days == 365
    assert years == 1
    assert months > 0
    assert weeks > 0


def test_calculate_time_span_no_dates():
    df = pd.DataFrame({
        "DateTimeOriginal": [None, None],
    })
    assert calculate_time_span(df) == (0, 0, 0, 0)


def test_calculate_time_span_same_day():
    df = pd.DataFrame({
        "DateTimeOriginal": pd.to_datetime(["2022-03-15", "2022-03-15"]),
    })
    years, months, weeks, days = calculate_time_span(df)
    assert days == 0
    assert years == 0


# --- overlay_logo ---

def test_overlay_logo_pastes_logo(tmp_path):
    # Create a main image and a logo
    main_img = Image.new("RGB", (200, 200), color="white")
    main_path = tmp_path / "chart.png"
    main_img.save(str(main_path))

    logo = Image.new("RGBA", (30, 30), color=(255, 0, 0, 128))

    overlay_logo(str(main_path), logo)

    # Verify the file was modified
    result = Image.open(str(main_path))
    assert result.size == (200, 200)
    # Top-right corner should have some red from the logo
    pixel = result.getpixel((185, 15))
    assert pixel[0] > 100  # Red channel should be significant


# --- fig2img (requires matplotlib) ---

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")
def test_fig2img_converts_figure():
    from addaxai.analysis.plots import fig2img

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    result = fig2img(fig)
    plt.close(fig)

    assert isinstance(result, Image.Image)
    assert result.size[0] > 0
    assert result.size[1] > 0
