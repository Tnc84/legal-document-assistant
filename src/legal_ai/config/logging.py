"""Logging configuration using loguru with structured fields."""

from __future__ import annotations

import logging
import sys
from typing import Any

from loguru import logger

from legal_ai.observability.context import get_request_id

_CONFIGURED = False

_TEXT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>req={extra[request_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _patch_record(record: dict[str, Any]) -> None:
    """Inject the current request id into every record's extra fields."""

    record["extra"].setdefault("request_id", get_request_id())


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


def configure_logging(level: str = "INFO", log_format: str = "text") -> None:
    """Configure loguru sinks once and bridge stdlib logging to it.

    Args:
        level: minimum log level for all sinks.
        log_format: ``"text"`` for human-readable output or ``"json"`` for
            structured one-line-per-record output suitable for log aggregation.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.configure(patcher=_patch_record)
    logger.remove()
    use_json = log_format.lower() == "json"
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        serialize=use_json,
        format="{message}" if use_json else _TEXT_FORMAT,
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
