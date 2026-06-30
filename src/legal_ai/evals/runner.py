"""RAG evaluation orchestrator.

Reuses the production components without reimplementing them:
``IngestionPipeline`` for indexing eval contracts, ``HybridRetriever`` for the
retrieval suite and ``QAChain`` for the QA suite. The LLM-backed QA chain is
built lazily so the retrieval suite can run without Ollama.
"""

from __future__ import annotations

from pathlib import Path

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings
from legal_ai.evals import answer_metrics, citation_metrics, retrieval_metrics
from legal_ai.evals.datasets import QACase, RetrievalCase
from legal_ai.inference.qa_chain import QAChain
from legal_ai.ingestion.embedder import Embedder
from legal_ai.ingestion.pipeline import IngestionPipeline, IngestionResult
from legal_ai.retrieval.hybrid_retriever import HybridRetriever
from legal_ai.retrieval.vector_store import QdrantVectorStore

_logger = get_logger("evals.runner")


class EvalRunner:
    """Ingest eval contracts and score retrieval / QA suites."""

    def __init__(
        self,
        settings: Settings,
        pipeline: IngestionPipeline | None = None,
        retriever: HybridRetriever | None = None,
        qa_chain: QAChain | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = Embedder(settings)
        self._vector_store = QdrantVectorStore(settings)
        self._pipeline = pipeline or IngestionPipeline(
            embedder=self._embedder, vector_store=self._vector_store, settings=settings
        )
        self._retriever = retriever or HybridRetriever(self._vector_store, self._embedder, settings)
        self._qa_chain = qa_chain

    def ingest_contracts(self, contracts_dir: str | Path) -> list[IngestionResult]:
        """Ingest every PDF in ``contracts_dir`` using its file stem as id."""

        directory = Path(contracts_dir)
        pdf_paths = sorted(directory.glob("*.pdf"))
        if not pdf_paths:
            _logger.warning(f"No eval contracts found in {directory}")
            return []
        results = [self._pipeline.ingest_pdf(path, document_id=path.stem) for path in pdf_paths]
        _logger.info(f"Ingested {len(results)} eval contracts from {directory}")
        return results

    def run_retrieval(
        self, cases: list[RetrievalCase], top_k: int
    ) -> retrieval_metrics.RetrievalSummary:
        """Score the retrieval suite (no LLM required)."""

        self._vector_store.ensure_collection()
        results = []
        for case in cases:
            retrieved = self._retriever.retrieve(
                case.query, top_k=top_k, document_ids=case.document_ids or None
            )
            hits = [chunk.hit for chunk in retrieved]
            results.append(retrieval_metrics.evaluate_retrieval_case(case, hits, top_k))
        return retrieval_metrics.summarize_retrieval(results, top_k)

    def run_qa(
        self, cases: list[QACase], top_k: int
    ) -> tuple[answer_metrics.AnswerSummary, citation_metrics.CitationSummary]:
        """Score the QA suite: answer keyword coverage + citation accuracy."""

        self._vector_store.ensure_collection()
        chain = self._require_qa_chain()
        answer_results = []
        citation_results = []
        for case in cases:
            response = chain.answer(
                case.question, document_ids=case.document_ids or None, top_k=top_k
            )
            answer_results.append(answer_metrics.evaluate_answer_case(case, response.answer))
            citation_results.append(
                citation_metrics.evaluate_citation_case(case, response.citations)
            )
        return (
            answer_metrics.summarize_answers(answer_results),
            citation_metrics.summarize_citations(citation_results),
        )

    def _require_qa_chain(self) -> QAChain:
        if self._qa_chain is None:
            self._qa_chain = self._build_qa_chain()
        return self._qa_chain

    def _build_qa_chain(self) -> QAChain:
        from legal_ai.inference.llm_client import OllamaClient
        from legal_ai.resilience.resilient_llm import ResilientLLMClient

        client = OllamaClient(self._settings)
        if self._settings.ollama_cb_enabled:
            llm = ResilientLLMClient(client, self._settings)
            return QAChain(self._retriever, llm, self._settings)
        return QAChain(self._retriever, client, self._settings)
