"""Embedding service using HuggingFace sentence-transformers."""

from __future__ import annotations

from typing import Sequence

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.observability.telemetry import get_tracer

_logger = get_logger("ingestion.embedder")
_tracer = get_tracer("legal_ai.ingestion.embedder")

_E5_PASSAGE_PREFIX = "passage: "
_E5_QUERY_PREFIX = "query: "


class Embedder:
    """Wrapper over sentence-transformers with E5-style prefixing."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        _logger.info(
            f"Loading embedding model {self._settings.embedding_model} on {self._settings.embedding_device}"
        )
        self._model = SentenceTransformer(
            self._settings.embedding_model,
            device=self._settings.embedding_device,
        )

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, _E5_PASSAGE_PREFIX)

    def encode_query(self, text: str) -> list[float]:
        vectors = self._encode([text], _E5_QUERY_PREFIX)
        return vectors[0]

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_model()
        prefixed = [f"{prefix}{text}" for text in texts]
        with _tracer.start_as_current_span("embed.encode") as span:
            span.set_attribute("embed.count", len(prefixed))
            span.set_attribute("embed.batch_size", self._settings.embedding_batch_size)
            embeddings = self._model.encode(  # type: ignore[union-attr]
                prefixed,
                batch_size=self._settings.embedding_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return [vector.tolist() for vector in embeddings]

    @property
    def dimension(self) -> int:
        return self._settings.embedding_dim
