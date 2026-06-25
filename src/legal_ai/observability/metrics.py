"""Application metric instruments for RAG operations and LLM token usage.

Instruments are created lazily from the global meter so they become no-ops when
metrics are disabled, and so this module never forces telemetry setup on import.
"""

from __future__ import annotations

from collections.abc import Iterable

from opentelemetry import metrics
from opentelemetry.metrics import (
    CallbackOptions,
    Counter,
    Histogram,
    ObservableGauge,
    Observation,
)

_METER_NAME = "legal_ai"

_llm_tokens: Counter | None = None
_operation_duration: Histogram | None = None
_cb_state_gauge: ObservableGauge | None = None
_cb_states: dict[str, int] = {}


def _ensure_instruments() -> None:
    global _llm_tokens, _operation_duration
    if _llm_tokens is not None and _operation_duration is not None:
        return
    meter = metrics.get_meter(_METER_NAME)
    _llm_tokens = meter.create_counter(
        name="llm.tokens",
        unit="token",
        description="LLM tokens consumed, split by direction (prompt/completion).",
    )
    _operation_duration = meter.create_histogram(
        name="rag.operation.duration",
        unit="s",
        description="Duration of core RAG operations (qa, risk, compare, retrieve, embed).",
    )


def record_llm_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record prompt and completion token counts for one LLM call."""

    _ensure_instruments()
    assert _llm_tokens is not None
    if prompt_tokens:
        _llm_tokens.add(prompt_tokens, {"model": model, "direction": "prompt"})
    if completion_tokens:
        _llm_tokens.add(completion_tokens, {"model": model, "direction": "completion"})


def record_operation(operation: str, duration_s: float, success: bool) -> None:
    """Record the wall-clock duration and outcome of a core operation."""

    _ensure_instruments()
    assert _operation_duration is not None
    _operation_duration.record(
        duration_s, {"operation": operation, "success": str(success).lower()}
    )


def _observe_circuit_breaker_state(
    options: CallbackOptions,
) -> Iterable[Observation]:
    return [Observation(value, {"breaker": name}) for name, value in _cb_states.items()]


def _ensure_cb_gauge() -> None:
    global _cb_state_gauge
    if _cb_state_gauge is not None:
        return
    meter = metrics.get_meter(_METER_NAME)
    _cb_state_gauge = meter.create_observable_gauge(
        name="circuit_breaker.state",
        callbacks=[_observe_circuit_breaker_state],
        description="Circuit breaker state (0=closed, 1=open, 2=half_open).",
    )


def record_circuit_breaker_state(name: str, state_value: int) -> None:
    """Publish the current state of a named circuit breaker as a gauge value."""

    _cb_states[name] = state_value
    _ensure_cb_gauge()
