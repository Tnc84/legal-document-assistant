"""Request-scoped context shared across logging and telemetry.

This module deliberately imports nothing from the rest of the package so it can
be used by the logging configuration without creating circular imports.
"""

from __future__ import annotations

from contextvars import ContextVar

_NO_REQUEST = "-"

_request_id_var: ContextVar[str] = ContextVar("request_id", default=_NO_REQUEST)


def set_request_id(request_id: str) -> None:
    """Bind the current request id to the active context."""

    _request_id_var.set(request_id)


def get_request_id() -> str:
    """Return the current request id or a placeholder when outside a request."""

    return _request_id_var.get()


def reset_request_id() -> None:
    """Clear the request id from the active context."""

    _request_id_var.set(_NO_REQUEST)
