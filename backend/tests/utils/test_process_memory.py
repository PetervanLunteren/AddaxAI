"""rss_mb is diagnostics-only: it must return a sane value on the
platforms CI runs on, and never raise."""

from app.utils.process_memory import rss_mb


def test_rss_mb_returns_positive_float():
    value = rss_mb()
    assert value is not None
    assert isinstance(value, float)
    # A running pytest process holds well over 10 MB and well under a TB.
    assert 10 < value < 1_000_000
