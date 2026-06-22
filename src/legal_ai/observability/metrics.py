"""Application metric instruments for RAG operations and LLM token usage.

Instruments are created lazily from the global meter so they become no-ops when
metrics are disabled, and so this module never forces telemetry setup on import.
"""

from __future__ import annotations

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram

_METER_NAME = "legal_ai"

_llm_tokens: Counter | None = None
_operation_duration: Histogram | None = None


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
