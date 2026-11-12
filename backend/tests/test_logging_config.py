"""
What `setup_logging` promises: the firehoses stay pinned below the
root's DEBUG, and console lines go to stdout.

The root stays at DEBUG for the beta so diagnostic reports carry
everything; the week of history that promises only holds while the
known firehoses are pinned. Pillow's TIFF plugin was 87% of a tester's
log before its pin (2026-09-06).
"""

import logging
import sys
from collections.abc import Iterator

import pytest

from app.core.logging_config import setup_logging


@pytest.fixture
def root_logger() -> Iterator[logging.Logger]:
    """The root logger after `setup_logging`, put back as it was afterwards."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_hook = sys.excepthook
    try:
        setup_logging()
        yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        sys.excepthook = saved_hook


def test_setup_logging_pins_the_firehoses_below_the_root_level(
    root_logger: logging.Logger,
) -> None:
    assert root_logger.level == logging.DEBUG
    assert logging.getLogger("PIL").level == logging.INFO
    assert logging.getLogger("PIL.TiffImagePlugin").getEffectiveLevel() == logging.INFO
    assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING


def test_console_lines_go_to_stdout_so_electron_does_not_label_them_errors(
    root_logger: logging.Logger,
) -> None:
    # Electron prints the backend's stderr as "[Backend Error]". A user
    # pasted INFO lines under that label as errors (forum, 2026-10-29).
    # The exact type: the file handler and pytest's own capture handlers
    # are StreamHandler subclasses.
    console_handlers = [
        handler
        for handler in root_logger.handlers
        if type(handler) is logging.StreamHandler
    ]
    assert [handler.stream for handler in console_handlers] == [sys.stdout]
