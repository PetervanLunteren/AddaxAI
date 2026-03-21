"""Logging setup for AddaxAI."""
import logging
import os
import sys
from typing import Optional


def setup_logging(log_dir: str = "", level: int = logging.INFO) -> None:
    """Configure the addaxai root logger with console + optional file handler.

    Args:
        log_dir: Directory for log file. If empty, file logging is disabled.
        level: Logging level (default INFO).
    """
    root_logger = logging.getLogger("addaxai")
    root_logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # File handler (optional)
    if log_dir and os.path.isdir(log_dir):
        fh = logging.FileHandler(
            os.path.join(log_dir, "addaxai.log"),
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root_logger.addHandler(fh)
