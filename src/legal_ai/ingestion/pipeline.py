"""End-to-end ingestion pipeline: PDF -> chunks -> embeddings -> Qdrant."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.ingestion.chunker import Chunk, SemanticChunker
from legal_ai.ingestion.embedder import Embedder
from legal_ai.ingestion.parser import PdfParser
from legal_ai.retrieval.vector_store import QdrantVectorStore

_logger = get_logger("ingestion.pipeline")


@dataclass(frozen=True)
class IngestionResult:
    """Summary of a single document ingestion run."""

    document_id: str
    title: str
    page_count: int
    chunk_count: int

    def to_dict(self) -> dict[str, str | int]:
        """Serialize to a JSON-friendly mapping (shared by API and worker)."""

        return {
            "document_id": self.document_id,
            "title": self.title,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
        }


class IngestionPipeline:
    """Orchestrate parsing, chunking, embedding and indexing for one document."""

    def __init__(
        self,
        parser: PdfParser | None = None,
        chunker: SemanticChunker | None = None,
        embedder: Embedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._parser = parser or PdfParser(extract_tables=True)
        self._chunker = chunker or SemanticChunker(
            max_tokens=self._settings.chunk_size_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
        )
        self._embedder = embedder or Embedder(self._settings)
        self._vector_store = vector_store or QdrantVectorStore(self._settings)

    def ingest_pdf(self, pdf_path: str | Path, document_id: str | None = None) -> IngestionResult:
        path = Path(pdf_path)
        document = self._parser.parse(path)
        doc_id = document_id or self._compute_document_id(path, document.full_text)

        chunks = self._chunker.chunk(document, doc_id)
        if not chunks:
            _logger.warning(f"No extractable content in {path}")
            return IngestionResult(
                document_id=doc_id,
                title=document.title,
                page_count=document.page_count,
                chunk_count=0,
            )

        self._vector_store.delete_document(doc_id)
        embeddings = self._embedder.encode_passages([chunk.text for chunk in chunks])
        self._vector_store.upsert_chunks(chunks, embeddings)

        _logger.info(f"Ingested document {document.title} (id={doc_id}) -> {len(chunks)} chunks")
        return IngestionResult(
            document_id=doc_id,
            title=document.title,
            page_count=document.page_count,
            chunk_count=len(chunks),
        )

    def chunk_document_only(
        self, pdf_path: str | Path, document_id: str | None = None
    ) -> list[Chunk]:
        path = Path(pdf_path)
        document = self._parser.parse(path)
        doc_id = document_id or self._compute_document_id(path, document.full_text)
        return self._chunker.chunk(document, doc_id)

    @staticmethod
    def _compute_document_id(path: Path, full_text: str) -> str:
        digest = hashlib.sha1()
        digest.update(path.name.encode("utf-8"))
        digest.update(full_text[:4096].encode("utf-8", errors="ignore"))
        return digest.hexdigest()[:16]
