"""OpenTelemetry bootstrap: tracer + meter providers and auto-instrumentation.

Keeps a single source of truth for telemetry setup so inference and retrieval
modules only depend on the small `get_tracer` helper.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings
from legal_ai.observability.metrics import record_operation

if TYPE_CHECKING:
    from fastapi import FastAPI

_logger = get_logger("observability.telemetry")

_INSTRUMENTATION_SCOPE = "legal_ai"

_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None
_configured = False


def configure_telemetry(settings: Settings, app: FastAPI) -> None:
    """Set up tracing and metrics providers once and instrument the app."""

    global _tracer_provider, _meter_provider, _configured
    if _configured:
        return

    resource = Resource.create({"service.name": settings.otel_service_name})

    if settings.otel_traces_enabled:
        _tracer_provider = TracerProvider(resource=resource)
        if settings.otel_exporter_otlp_endpoint:
            exporter = OTLPSpanExporter(
                endpoint=f"{settings.otel_exporter_otlp_endpoint.rstrip('/')}/v1/traces"
            )
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            _logger.info(f"OTLP trace export enabled -> {settings.otel_exporter_otlp_endpoint}")
        trace.set_tracer_provider(_tracer_provider)

    if settings.metrics_enabled:
        _meter_provider = MeterProvider(
            resource=resource, metric_readers=[PrometheusMetricReader()]
        )
        metrics.set_meter_provider(_meter_provider)
        _logger.info("Prometheus metrics enabled at /metrics")

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()

    _configured = True
    _logger.info(f"Telemetry configured for service {settings.otel_service_name}")


def get_tracer(name: str = _INSTRUMENTATION_SCOPE) -> Tracer:
    """Return a tracer scoped to the given instrumentation name."""

    return trace.get_tracer(name)


@contextmanager
def traced_operation(operation: str) -> Iterator[trace.Span]:
    """Trace a core operation and record its duration and outcome as a metric.

    Args:
        operation: short metric label (e.g. ``"qa"``); the span is named
            ``"rag.<operation>"``.
    """

    start = perf_counter()
    success = False
    with get_tracer(_INSTRUMENTATION_SCOPE).start_as_current_span(f"rag.{operation}") as span:
        try:
            yield span
            success = True
        finally:
            record_operation(operation, perf_counter() - start, success)


def shutdown_telemetry() -> None:
    """Flush and shut down providers during graceful shutdown."""

    global _configured
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
    if _meter_provider is not None:
        _meter_provider.shutdown()
    _configured = False
    _logger.info("Telemetry shut down")
