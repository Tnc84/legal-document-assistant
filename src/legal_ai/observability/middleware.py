"""ASGI middleware binding a request id and recording access timing."""

from __future__ import annotations

import time
import uuid

from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from legal_ai.config.logging import get_logger
from legal_ai.observability.context import reset_request_id, set_request_id

_logger = get_logger("observability.request")

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, time the request and emit a structured access log."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        set_request_id(request_id)

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("request.id", request_id)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[_REQUEST_ID_HEADER] = request_id
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            _logger.bind(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                latency_ms=round(latency_ms, 2),
            ).info(f"{request.method} {request.url.path} -> {status_code}")
            reset_request_id()
