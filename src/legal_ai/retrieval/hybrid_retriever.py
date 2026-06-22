"""Hybrid retriever fusing dense (Qdrant) and BM25 (rank-bm25)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.ingestion.embedder import Embedder
from legal_ai.observability.telemetry import get_tracer
from legal_ai.retrieval.vector_store import QdrantVectorStore, SearchHit

_logger = get_logger("retrieval.hybrid")
_tracer = get_tracer("legal_ai.retrieval.hybrid")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk plus the fused hybrid score used for ranking."""

    hit: SearchHit
    dense_score: float
    bm25_score: float
    fused_score: float


class HybridRetriever:
    """Combine dense vector search with BM25 over the candidate pool.

    The candidate pool is taken from a wider Qdrant search and re-ranked with BM25
    on the chunk text. This keeps memory bounded and avoids holding a global BM25
    index in process state.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        effective_top_k = top_k or self._settings.retrieval_top_k
        candidate_k = max(effective_top_k * 4, 20)

        with _tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("retrieval.top_k", effective_top_k)
            span.set_attribute("retrieval.candidate_k", candidate_k)

            query_vector = self._embedder.encode_query(query)
            dense_hits = self._vector_store.search(
                query_vector=query_vector,
                top_k=candidate_k,
                document_ids=document_ids,
            )
            if not dense_hits:
                span.set_attribute("retrieval.hits_count", 0)
                return []

            bm25_scores = self._bm25_scores(query, dense_hits)
            dense_norm = self._min_max_normalize([hit.score for hit in dense_hits])
            bm25_norm = self._min_max_normalize(bm25_scores)

            results: list[RetrievedChunk] = []
            for hit, dense, bm25 in zip(dense_hits, dense_norm, bm25_norm):
                fused = (
                    self._settings.retrieval_dense_weight * dense
                    + self._settings.retrieval_bm25_weight * bm25
                )
                results.append(
                    RetrievedChunk(
                        hit=hit,
                        dense_score=dense,
                        bm25_score=bm25,
                        fused_score=fused,
                    )
                )
            results.sort(key=lambda item: item.fused_score, reverse=True)
            top_results = results[:effective_top_k]
            span.set_attribute("retrieval.hits_count", len(top_results))
            return top_results

    @staticmethod
    def _bm25_scores(query: str, hits: list[SearchHit]) -> list[float]:
        corpus_tokens = [_TOKEN_RE.findall(hit.text.lower()) for hit in hits]
        if not any(corpus_tokens):
            return [0.0] * len(hits)
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = _TOKEN_RE.findall(query.lower())
        scores = bm25.get_scores(query_tokens)
        return [float(score) for score in scores]

    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return [0.0 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]
