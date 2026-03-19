"""Plotting and visualization utilities for AddaxAI.

Chart helpers, figure conversion, and logo overlay.
Heavy plotting functions (time-series, maps, heatmaps) remain in the
main produce_plots() orchestrator for now and will be extracted in a
later phase.
"""

import io
import os

from PIL import Image


def fig2img(fig):
    """Convert a matplotlib figure to a PIL Image via in-memory buffer.

    Args:
        fig: A matplotlib figure or pyplot module.

    Returns:
        PIL.Image.Image
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


def overlay_logo(image_path, logo):
    """Paste a logo image onto the top-right corner of a chart image.

    Args:
        image_path: Path to the chart image (modified in place).
        logo: PIL.Image.Image of the logo to overlay.
    """
    main_image = Image.open(image_path)
    main_width, main_height = main_image.size
    logo_width, logo_height = logo.size
    position = (main_width - logo_width - 10, 10)
    main_image.paste(logo, position, logo)
    main_image.save(image_path)


def calculate_time_span(df):
    """Analyze the date range in a detection DataFrame.

    Args:
        df: pandas DataFrame with a 'DateTimeOriginal' datetime column.

    Returns:
        Tuple of (years, months, weeks, days) as integers.
        Returns (0, 0, 0, 0) if no dates are present.
    """
    any_dates_present = df['DateTimeOriginal'].notnull().any()
    if not any_dates_present:
        return 0, 0, 0, 0
    first_date = df['DateTimeOriginal'].min()
    last_date = df['DateTimeOriginal'].max()
    time_difference = last_date - first_date
    days = time_difference.days
    years = int(days / 365)
    months = int(days / 30)
    weeks = int(days / 7)
    return years, months, weeks, days
