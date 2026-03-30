"""
Centralized logging configuration for the SUT Evaluation Framework.

Provides a configured logger with console and file handlers (rotating).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Constants
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "sut_eval.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str = "sut_eval",
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Setup and return a configured logger.

    Args:
        name: Logger name (default: "sut_eval")
        log_level: Logging level (default: logging.INFO)
        log_to_file: Whether to write logs to file (default: True)
        log_to_console: Whether to write logs to console (default: True)

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Don't add handlers if they already exist (prevent duplicate logs)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    # File Handler
    if log_to_file:
        # Ensure log directory exists
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG in file
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback if file logging fails (e.g. permissions)
            print(f"Failed to setup file logging: {e}", file=sys.stderr)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    If name is None, returns the root 'sut_eval' logger.
    If name is provided, returns a child logger (e.g. 'sut_eval.module').
    """
    if name:
        # Create a child logger if name is provided (e.g., 'runner')
        # Ensure it propagates to the parent 'sut_eval' logger
        return logging.getLogger(f"sut_eval.{name}")
    return logging.getLogger("sut_eval")
