"""
The loggers `setup_logging` pins below the root's DEBUG.

The root stays at DEBUG for the beta so diagnostic reports carry
everything; the week of history that promises only holds while the
known firehoses are pinned. Pillow's TIFF plugin was 87% of a tester's
log before its pin (2026-09-06).
"""

import logging
import sys

from app.core.logging_config import setup_logging


def test_setup_logging_pins_the_firehoses_below_the_root_level():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_hook = sys.excepthook
    try:
        setup_logging()
        assert root.level == logging.DEBUG
        assert logging.getLogger("PIL").level == logging.INFO
        assert logging.getLogger("PIL.TiffImagePlugin").getEffectiveLevel() == logging.INFO
        assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        sys.excepthook = saved_hook
