"""RAG Q&A chain with source attribution."""

from __future__ import annotations

from dataclasses import dataclass

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.inference.llm_client import LLMClient, load_prompt
from legal_ai.observability.telemetry import traced_operation
from legal_ai.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk

_logger = get_logger("inference.qa")


@dataclass(frozen=True)
class Citation:
    """Citation metadata returned alongside the answer."""

    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section_path: str
    snippet: str
    score: float


@dataclass(frozen=True)
class QAResponse:
    """Q&A pipeline output."""

    question: str
    answer: str
    citations: list[Citation]


class QAChain:
    """Retrieve hybrid context and generate a grounded answer with citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or get_settings()
        self._system_prompt = load_prompt("qa_system")

    def answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> QAResponse:
        with traced_operation("qa") as span:
            span.set_attribute("qa.document_ids", document_ids or [])
            retrieved = self._retriever.retrieve(
                question, top_k=top_k, document_ids=document_ids
            )
            if not retrieved:
                span.set_attribute("qa.retrieved", 0)
                return QAResponse(
                    question=question,
                    answer="No indexed content matches this question.",
                    citations=[],
                )

            span.set_attribute("qa.retrieved", len(retrieved))
            user_prompt = self._build_user_prompt(question, retrieved)
            raw_answer = self._llm.complete(self._system_prompt, user_prompt)
            citations = [self._to_citation(item) for item in retrieved]
            return QAResponse(
                question=question, answer=raw_answer.strip(), citations=citations
            )

    @staticmethod
    def _build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
        context_blocks: list[str] = []
        for index, item in enumerate(retrieved, start=1):
            hit = item.hit
            header = (
                f"[#{index} | doc={hit.document_title} | pages={hit.page_start}-{hit.page_end} | "
                f"section={hit.section_path or 'N/A'} | score={item.fused_score:.3f}]"
            )
            context_blocks.append(f"{header}\n{hit.text}")
        joined_context = "\n\n---\n\n".join(context_blocks)
        return (
            f"Question:\n{question}\n\n"
            f"Contract excerpts:\n{joined_context}\n\n"
            "Answer using only these excerpts and cite them inline as "
            "[doc=..., page=..., section=...]."
        )

    @staticmethod
    def _to_citation(item: RetrievedChunk) -> Citation:
        hit = item.hit
        snippet = hit.text[:600] + ("..." if len(hit.text) > 600 else "")
        return Citation(
            document_id=hit.document_id,
            document_title=hit.document_title,
            page_start=hit.page_start,
            page_end=hit.page_end,
            section_path=hit.section_path,
            snippet=snippet,
            score=item.fused_score,
        )
