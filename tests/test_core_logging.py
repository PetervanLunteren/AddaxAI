"""Tests for addaxai.core.logging (Phase 5.2a)."""
import logging

from addaxai.core.logging import setup_logging


def test_setup_logging_is_importable():
    assert callable(setup_logging)


def test_setup_logging_adds_handler():
    logger = logging.getLogger("addaxai")
    # Remove any handlers left from a prior call in this process
    logger.handlers.clear()
    setup_logging()
    assert len(logger.handlers) >= 1


def test_setup_logging_info_level(capsys):
    logger = logging.getLogger("addaxai")
    logger.handlers.clear()
    setup_logging(level=logging.INFO)
    logger.info("test-log-message-xyz")
    captured = capsys.readouterr()
    assert "test-log-message-xyz" in captured.out


def test_setup_logging_debug_suppressed_at_info(capsys):
    logger = logging.getLogger("addaxai")
    logger.handlers.clear()
    setup_logging(level=logging.INFO)
    logger.debug("should-not-appear")
    captured = capsys.readouterr()
    assert "should-not-appear" not in captured.out


def test_setup_logging_no_file_when_dir_empty(tmp_path):
    """With empty log_dir, no file handler should be added."""
    logger = logging.getLogger("addaxai")
    logger.handlers.clear()
    setup_logging(log_dir="")
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 0


def test_setup_logging_file_handler_created(tmp_path):
    """With a valid log_dir, a FileHandler should be added."""
    logger = logging.getLogger("addaxai")
    logger.handlers.clear()
    setup_logging(log_dir=str(tmp_path))
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    log_file = tmp_path / "addaxai.log"
    assert log_file.exists()
