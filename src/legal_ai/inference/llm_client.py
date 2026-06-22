"""LLM client abstraction backed by Ollama.

Defines a small `LLMClient` Protocol so we can swap providers without touching
inference modules. The Ollama implementation uses the official `/api/chat`
endpoint and supports JSON-mode for structured outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.observability.metrics import record_llm_tokens
from legal_ai.observability.telemetry import get_tracer

_tracer = get_tracer("legal_ai.inference.llm")

_logger = get_logger("inference.llm")
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


class LLMClient(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


class OllamaClient:
    """Concrete `LLMClient` calling a local Ollama daemon."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = httpx.Client(
            base_url=self._settings.ollama_host,
            timeout=self._settings.ollama_request_timeout,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1.0, max=8.0),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
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
        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temperature if temperature is not None else self._settings.ollama_temperature,
                "num_predict": max_tokens or self._settings.ollama_max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        model = self._settings.ollama_model
        _logger.debug(f"Ollama call model={model} json_mode={json_mode}")
        with _tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.json_mode", json_mode)
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            message = data.get("message") or {}
            content = message.get("content", "")
            if not isinstance(content, str):
                raise ValueError(f"Unexpected Ollama response shape: {data}")
            self._record_usage(span, model, data)
            return content

    @staticmethod
    def _record_usage(span: Any, model: str, data: dict[str, Any]) -> None:
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        span.set_attribute("llm.prompt_tokens", prompt_tokens)
        span.set_attribute("llm.completion_tokens", completion_tokens)
        record_llm_tokens(model, prompt_tokens, completion_tokens)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        raw = self.complete(
            system_prompt,
            user_prompt,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return safe_json_loads(raw)

    def close(self) -> None:
        self._client.close()


def load_prompt(name: str) -> str:
    """Load a prompt file from the project `prompts/` directory."""

    candidates = [
        _PROMPTS_DIR / f"{name}.md",
        Path("prompts") / f"{name}.md",
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"Prompt {name} not found in {[str(p) for p in candidates]}")


def safe_json_loads(raw: str) -> dict[str, Any]:
    """Tolerant JSON parsing: strip code fences and extract first object."""

    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return JSON: {raw[:200]}")
    snippet = text[start : end + 1]
    return json.loads(snippet)
