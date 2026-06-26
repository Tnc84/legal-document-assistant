"""Generic synchronous circuit breaker.

Protects a downstream dependency (e.g. the Ollama daemon) from repeated calls
while it is failing. Three states:

- ``CLOSED``: calls pass through; consecutive failures are counted.
- ``OPEN``: calls fail fast with :class:`CircuitBreakerError` until the recovery
  timeout elapses.
- ``HALF_OPEN``: a single trial call is allowed; success closes the circuit,
  failure re-opens it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum
from time import monotonic
from typing import TypeVar

from legal_ai.config.logging import get_logger

_logger = get_logger("resilience.circuit_breaker")

T = TypeVar("T")


class CircuitState(StrEnum):
    """Lifecycle states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


StateListener = Callable[[str, CircuitState], None]


class CircuitBreakerError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = max(0, int(round(retry_after)))
        super().__init__(f"Circuit breaker '{name}' is open; retry after {self.retry_after}s")


class CircuitBreaker:
    """Thread-safe circuit breaker wrapping arbitrary synchronous callables."""

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        recovery_timeout: float,
        state_listener: StateListener | None = None,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state_listener = state_listener
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._notify()

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        return self._state

    def call(self, func: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Execute ``func`` through the breaker, enforcing the current state."""

        self._before_call()
        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def _before_call(self) -> None:
        with self._lock:
            if self._state is not CircuitState.OPEN:
                return
            elapsed = monotonic() - self._opened_at
            if elapsed < self._recovery_timeout:
                raise CircuitBreakerError(self._name, self._recovery_timeout - elapsed)
            self._transition(CircuitState.HALF_OPEN)
            _logger.info(f"Circuit breaker '{self._name}' entering half-open probe")

    def _on_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state is not CircuitState.CLOSED:
                self._transition(CircuitState.CLOSED)
                _logger.info(f"Circuit breaker '{self._name}' closed after success")

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            tripped = self._failure_count >= self._failure_threshold
            if self._state is CircuitState.HALF_OPEN or tripped:
                self._open()

    def _open(self) -> None:
        self._opened_at = monotonic()
        self._transition(CircuitState.OPEN)
        _logger.warning(
            f"Circuit breaker '{self._name}' opened after {self._failure_count} failures"
        )

    def _transition(self, new_state: CircuitState) -> None:
        self._state = new_state
        self._notify()

    def _notify(self) -> None:
        if self._state_listener is not None:
            self._state_listener(self._name, self._state)
