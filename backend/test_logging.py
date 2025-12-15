"""
Test logging configuration and basic functionality.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- No silent failures
"""

import logging
import tempfile
from pathlib import Path

import pytest

from app.core.logging_config import get_logger, setup_logging


def test_setup_logging_creates_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that setup_logging creates the log file."""
    # Create a temporary user data directory
    user_data_dir = tmp_path / "AddaxAI"
    user_data_dir.mkdir()

    # Mock get_settings to return our temp directory
    from unittest.mock import Mock

    mock_settings = Mock()
    mock_settings.user_data_dir = user_data_dir
    mock_settings.environment = "test"
    mock_settings.debug = False

    monkeypatch.setattr("app.core.logging_config.get_settings", lambda: mock_settings)

    # Setup logging
    logger = setup_logging()

    # Verify log file was created
    log_file = user_data_dir / "logs" / "backend.log"
    assert log_file.exists(), "Log file should be created"

    # Verify we can write to it
    logger.info("Test message")

    # Read log file and verify content
    content = log_file.read_text()
    assert "Test message" in content
    assert "[INFO]" in content
    assert "[addaxai]" in content


def test_get_logger_creates_module_logger() -> None:
    """Test that get_logger creates a logger with the correct name."""
    logger = get_logger("test_module")

    assert logger.name == "addaxai.test_module"
    assert isinstance(logger, logging.Logger)


def test_all_log_levels_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that all log levels (info, warning, error, critical) work."""
    # Create temp directory
    user_data_dir = tmp_path / "AddaxAI"
    user_data_dir.mkdir()

    from unittest.mock import Mock

    mock_settings = Mock()
    mock_settings.user_data_dir = user_data_dir
    mock_settings.environment = "test"
    mock_settings.debug = False

    monkeypatch.setattr("app.core.logging_config.get_settings", lambda: mock_settings)

    # Setup logging
    setup_logging()
    logger = get_logger("test_levels")

    # Log at all levels
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Read log file
    log_file = user_data_dir / "logs" / "backend.log"
    content = log_file.read_text()

    # Verify all levels appear
    assert "[INFO]" in content and "Info message" in content
    assert "[WARNING]" in content and "Warning message" in content
    assert "[ERROR]" in content and "Error message" in content
    assert "[CRITICAL]" in content and "Critical message" in content


def test_exception_logging_includes_stack_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that logging exceptions includes stack traces."""
    # Create temp directory
    user_data_dir = tmp_path / "AddaxAI"
    user_data_dir.mkdir()

    from unittest.mock import Mock

    mock_settings = Mock()
    mock_settings.user_data_dir = user_data_dir
    mock_settings.environment = "test"
    mock_settings.debug = False

    monkeypatch.setattr("app.core.logging_config.get_settings", lambda: mock_settings)

    # Setup logging
    setup_logging()
    logger = get_logger("test_exception")

    # Create and log an exception
    try:
        raise ValueError("Test exception for logging")
    except ValueError as e:
        logger.error(f"Caught exception: {e}", exc_info=True)

    # Read log file
    log_file = user_data_dir / "logs" / "backend.log"
    content = log_file.read_text()

    # Verify exception details appear
    assert "ValueError: Test exception for logging" in content
    assert "Traceback" in content
    assert "raise ValueError" in content


def test_log_format_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that log entries have the correct format."""
    # Create temp directory
    user_data_dir = tmp_path / "AddaxAI"
    user_data_dir.mkdir()

    from unittest.mock import Mock

    mock_settings = Mock()
    mock_settings.user_data_dir = user_data_dir
    mock_settings.environment = "test"
    mock_settings.debug = False

    monkeypatch.setattr("app.core.logging_config.get_settings", lambda: mock_settings)

    # Setup logging
    setup_logging()
    logger = get_logger("test_format")

    # Log a message
    logger.info("Test format message")

    # Read log file
    log_file = user_data_dir / "logs" / "backend.log"
    content = log_file.read_text()

    # Verify format: [YYYY-MM-DD HH:MM:SS] [LEVEL] [module] message
    import re

    pattern = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] \[addaxai\.test_format\] Test format message"
    assert re.search(pattern, content), f"Log format doesn't match expected pattern. Content:\n{content}"


def test_log_rotation_configuration() -> None:
    """Test that log rotation is configured correctly."""
    from app.core.logging_config import get_logger

    logger = get_logger("test_rotation")

    # Get root logger (handlers are on the root addaxai logger)
    root_logger = logging.getLogger("addaxai")

    # Check that root logger has handlers
    assert len(root_logger.handlers) > 0, "Root logger should have handlers"

    # Find RotatingFileHandler
    from logging.handlers import RotatingFileHandler

    rotating_handlers = [h for h in root_logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(rotating_handlers) > 0, "Root logger should have a RotatingFileHandler"

    handler = rotating_handlers[0]

    # Verify rotation settings
    assert handler.maxBytes == 33 * 1024 * 1024, "Max bytes should be 33MB"
    assert handler.backupCount == 3, "Should keep 3 backups"
