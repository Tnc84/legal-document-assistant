"""Logging configuration using loguru with structured fields."""

from __future__ import annotations

import logging
import sys
from typing import Any

from loguru import logger

_CONFIGURED = False


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru sinks once and bridge stdlib logging to it."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx"):
        logging.getLogger(noisy).handlers = [_InterceptHandler()]
        logging.getLogger(noisy).propagate = False

    _CONFIGURED = True


def get_logger(name: str | None = None, **context: Any) -> Any:
    """Return a loguru logger bound to the given name and optional context."""

    bound = logger.bind(component=name) if name else logger
    if context:
        bound = bound.bind(**context)
    return bound
