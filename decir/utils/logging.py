"""Logging utilities for DeCIR."""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str = "decir",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Setup a logger with file and/or console handlers.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).
        log_file: Path to log file (optional).
        console: Whether to add console handler (default: True).

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger("my_experiment", level=logging.DEBUG, log_file="run.log")
        >>> logger.info("Starting experiment")
        >>> logger.debug("Debug information")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global default logger
default_logger = setup_logger()
