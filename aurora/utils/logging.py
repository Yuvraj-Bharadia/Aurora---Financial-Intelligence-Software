"""
Structured logging setup for Aurora using loguru.

Usage:
    from aurora.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Starting pipeline", ticker="SPY", horizon=5)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _logger


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    serialize: bool = False,
) -> None:
    """Configure loguru with console + optional file sinks."""
    _logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
    )
    _logger.add(sys.stderr, level=level, format=fmt, colorize=True, enqueue=True)

    if log_file is not None:
        _logger.add(
            str(log_file),
            level=level,
            rotation=rotation,
            retention=retention,
            serialize=serialize,
            enqueue=True,
        )


def get_logger(name: str) -> Any:
    """Return a logger bound with the given module name."""
    return _logger.bind(name=name)


# Apply defaults on import
configure_logging()
