"""Qdrant-backed vector store with payload-aware upsert and search."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.ingestion.chunker import Chunk

_logger = get_logger("retrieval.vector_store")


@dataclass(frozen=True)
class SearchHit:
    """One Qdrant search result with the payload required for citation."""

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page_start: int
    page_end: int
    section_path: str
    score: float


class QdrantVectorStore:
    """Thin abstraction over Qdrant with stable payload schema."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = QdrantClient(
            url=self._settings.qdrant_url,
            api_key=self._settings.qdrant_api_key or None,
            timeout=30.0,
        )
        self._collection = self._settings.qdrant_collection

    def ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return
        _logger.info(f"Creating Qdrant collection {self._collection}")
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=self._settings.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.ensure_collection()
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=self._build_payload(chunk),
            )
            for chunk, vector in zip(chunks, embeddings, strict=False)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        _logger.info(f"Upserted {len(points)} chunks into {self._collection}")

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        query_filter = self._build_filter(document_ids)
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [self._hit_from_point(point) for point in response.points]

    def fetch_document_chunks(self, document_id: str, limit: int = 1000) -> list[SearchHit]:
        flt = self._build_filter([document_id])
        records, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[SearchHit] = []
        for record in records:
            payload = record.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id", record.id)),
                    document_id=str(payload.get("document_id", "")),
                    document_title=str(payload.get("document_title", "")),
                    text=str(payload.get("text", "")),
                    page_start=int(payload.get("page_start", 0)),
                    page_end=int(payload.get("page_end", 0)),
                    section_path=str(payload.get("section_path", "")),
                    score=0.0,
                )
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        flt = self._build_filter([document_id])
        if flt is None:
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(filter=flt),
        )

    @staticmethod
    def _build_filter(document_ids: list[str] | None) -> qmodels.Filter | None:
        if not document_ids:
            return None
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchAny(any=document_ids),
                )
            ]
        )

    @staticmethod
    def _build_payload(chunk: Chunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "text": chunk.text,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_path": chunk.section_path,
            "token_estimate": chunk.token_estimate,
            **chunk.metadata,
        }

    @staticmethod
    def _hit_from_point(point: Any) -> SearchHit:
        payload = point.payload or {}
        return SearchHit(
            chunk_id=str(payload.get("chunk_id", point.id)),
            document_id=str(payload.get("document_id", "")),
            document_title=str(payload.get("document_title", "")),
            text=str(payload.get("text", "")),
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            section_path=str(payload.get("section_path", "")),
            score=float(point.score or 0.0),
        )
