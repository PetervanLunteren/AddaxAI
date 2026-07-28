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

from app.core.config import get_settings


def setup_logging() -> logging.Logger:
    """
    Set up application-wide logging.

    Creates log directory if it doesn't exist.
    Configures rotating file handler with 33MB rotation, 3 backups.

    IMPORTANT: Captures ALL logs including:
    - Application logs (addaxai.*)
    - Uvicorn server logs
    - SQLAlchemy logs
    - Third-party library logs
    - Uncaught exceptions and tracebacks

    Returns:
        Configured root logger
    """
    settings = get_settings()

    # Create logs directory
    logs_dir = settings.user_data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "backend.log"

    # Configure ROOT logger to capture EVERYTHING.
    #
    # Level is DEBUG during the beta phase so bug reports carry the
    # full picture: subprocess output, ML inference per-line traces,
    # SQLAlchemy parameter binds, HTTP request bodies. Diagnostic ZIPs
    # are the primary signal we have for testers' problems and we'd
    # rather have too much than too little. Drop back to INFO once
    # the app stabilises and beta winds down.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Rotating file handler (33MB per file, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=33 * 1024 * 1024,  # 33MB
        backupCount=3,
        encoding="utf-8",
    )

    # Format: [2024-12-15 10:30:45] [INFO] [module.name] Message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Also log to console in development
    if settings.debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Configure Uvicorn loggers to use our handler
    for uvicorn_logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True  # Propagate to root logger

    # SQLAlchemy logs one line per query (plus parameter binds) at INFO.
    # With the root logger at DEBUG that inheritance turns a single
    # analysis — tens of thousands of inserts — into a log firehose, and
    # because the file handler writes synchronously it stalls the
    # single-process server (request timeouts, sluggish UI). Pin the
    # SQLAlchemy loggers to WARNING so errors still surface but the
    # per-query noise is gone. Setting `sql_echo` re-enables the full
    # trace (create_engine(echo=True) forces these back to INFO).
    if not settings.sql_echo:
        for sa_logger_name in (
            "sqlalchemy.engine",
            "sqlalchemy.engine.Engine",
            "sqlalchemy.pool",
        ):
            logging.getLogger(sa_logger_name).setLevel(logging.WARNING)

    # Set up exception hook to log uncaught exceptions
    import sys

    def exception_handler(exc_type, exc_value, exc_traceback):
        """Log uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        root_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = exception_handler

    # Get app logger for initialization message
    logger = logging.getLogger("addaxai")
    logger.info("=" * 80)
    logger.info("AddaxAI Backend Logging Initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Environment: {settings.environment}")
    logger.info("Capturing ALL logs: uvicorn, SQLAlchemy, exceptions, tracebacks")
    logger.info("=" * 80)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (e.g., __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"addaxai.{name}")
