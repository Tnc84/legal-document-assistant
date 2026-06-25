"""Circuit-breaker-protected :class:`LLMClient` decorator.

Wraps any concrete ``LLMClient`` so that repeated Ollama failures trip a circuit
breaker and surface as :class:`CircuitBreakerError` instead of hammering an
unhealthy backend.
"""

from __future__ import annotations

from legal_ai.config.settings import Settings, get_settings
from legal_ai.inference.llm_client import LLMClient
from legal_ai.resilience.circuit_breaker import CircuitBreaker


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
