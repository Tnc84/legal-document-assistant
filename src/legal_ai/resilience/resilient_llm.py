"""Circuit-breaker-protected :class:`LLMClient` decorator.

Wraps any concrete ``LLMClient`` so that repeated Ollama failures trip a circuit
breaker and surface as :class:`CircuitBreakerError` instead of hammering an
unhealthy backend.
"""

from __future__ import annotations

from legal_ai.config.settings import Settings, get_settings
from legal_ai.inference.llm_client import LLMClient
from legal_ai.observability.metrics import record_circuit_breaker_state
from legal_ai.resilience.circuit_breaker import CircuitBreaker, CircuitState

_STATE_METRIC_VALUE: dict[CircuitState, int] = {
    CircuitState.CLOSED: 0,
    CircuitState.OPEN: 1,
    CircuitState.HALF_OPEN: 2,
}


def _record_breaker_metric(name: str, state: CircuitState) -> None:
    """Bridge circuit breaker state changes to the metrics layer."""

    record_circuit_breaker_state(name, _STATE_METRIC_VALUE[state])


class ResilientLLMClient:
    """Decorate an ``LLMClient`` with a circuit breaker around ``complete``."""

    def __init__(
        self,
        inner: LLMClient,
        settings: Settings | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._inner = inner
        self._breaker = breaker or CircuitBreaker(
            name="ollama",
            failure_threshold=resolved.ollama_cb_failure_threshold,
            recovery_timeout=resolved.ollama_cb_recovery_timeout,
            state_listener=_record_breaker_metric,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._breaker.call(
            self._inner.complete,
            system_prompt,
            user_prompt,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
