"""
Logging configuration for AddaxAI Backend.

Following DEVELOPERS.md principles:
- Explicit configuration
- Structured logging to files
- No silent failures

Sets up rotating file logging to ~/AddaxAI/logs/backend.log
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import get_settings


def setup_logging() -> logging.Logger:
    """
    Set up application-wide logging.

    Creates log directory if it doesn't exist.
    Configures rotating file handler with 33MB rotation, 3 backups.

    Returns:
        Configured root logger
    """
    settings = get_settings()

    # Create logs directory
    logs_dir = settings.user_data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "backend.log"

    # Configure root logger
    logger = logging.getLogger("addaxai")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Rotating file handler (33MB per file, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=33 * 1024 * 1024,  # 33MB
        backupCount=3,
        encoding="utf-8",
    )

    # Format: [2024-12-15 10:30:45] [INFO] [addaxai.main] Starting application
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Also log to console in development
    if settings.debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("AddaxAI Backend Logging Initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Environment: {settings.environment}")
    logger.info("=" * 80)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (e.g., __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"addaxai.{name}")
