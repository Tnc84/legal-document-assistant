"""Retrieval quality metrics: Recall@k, MRR and Hit@k.

Metrics are computed against a relevance "key set" derived from the ground
truth (see :class:`RelevanceJudge`). A retrieved chunk *covers* zero or more
keys; ``Recall@k`` is the fraction of ground-truth keys covered within the top
``k`` results, ``Hit@k`` is whether at least one key was covered, and the
reciprocal rank is ``1 / rank`` of the first covering chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from legal_ai.evals.datasets import RetrievalCase
from legal_ai.retrieval.vector_store import SearchHit


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Per-case retrieval scores."""

    case_id: str
    hit: bool
    recall_at_k: float
    reciprocal_rank: float
    relevant_total: int
    retrieved_total: int


@dataclass(frozen=True)
class RetrievalSummary:
    """Aggregated retrieval scores over all scorable cases."""

    top_k: int
    case_count: int
    recall_at_k: float
    mrr: float
    hit_rate: float
    per_case: list[RetrievalCaseResult] = field(default_factory=list)


class RelevanceJudge:
    """Map a retrieved chunk to the ground-truth relevance keys it covers."""

    def __init__(self, dimension: str, keys: frozenset[str]) -> None:
        self._dimension = dimension
        self._keys = keys

    @property
    def keys(self) -> frozenset[str]:
        return self._keys

    @classmethod
    def from_case(cls, case: RetrievalCase) -> RelevanceJudge:
        """Pick the highest-priority populated ground-truth signal."""

        if case.relevant_chunk_ids:
            return cls("chunk_id", frozenset(case.relevant_chunk_ids))
        if case.relevant_pages:
            return cls("page", frozenset(str(page) for page in case.relevant_pages))
        if case.relevant_section_substrings:
            return cls("section", frozenset(s.lower() for s in case.relevant_section_substrings))
        return cls("none", frozenset())

    def covered_keys(self, hit: SearchHit) -> frozenset[str]:
        if self._dimension == "chunk_id":
            return self._keys & {hit.chunk_id}
        if self._dimension == "page":
            pages = {str(page) for page in range(hit.page_start, hit.page_end + 1)}
            return self._keys & pages
        if self._dimension == "section":
            section = hit.section_path.lower()
            return frozenset(key for key in self._keys if key in section)
        return frozenset()


def evaluate_retrieval_case(
    case: RetrievalCase, hits: list[SearchHit], top_k: int
) -> RetrievalCaseResult:
    """Score a single retrieval case against the top-``k`` hits."""

    judge = RelevanceJudge.from_case(case)
    top_hits = hits[:top_k]
    covered: set[str] = set()
    first_relevant_rank: int | None = None
    for rank, hit in enumerate(top_hits, start=1):
        keys = judge.covered_keys(hit)
        if keys:
            covered |= set(keys)
            if first_relevant_rank is None:
                first_relevant_rank = rank

    total = len(judge.keys)
    recall = len(covered) / total if total else 0.0
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    return RetrievalCaseResult(
        case_id=case.id,
        hit=first_relevant_rank is not None,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        relevant_total=total,
        retrieved_total=len(top_hits),
    )


def summarize_retrieval(results: list[RetrievalCaseResult], top_k: int) -> RetrievalSummary:
    """Aggregate per-case results, ignoring cases without ground truth."""

    scored = [result for result in results if result.relevant_total > 0]
    if not scored:
        return RetrievalSummary(
            top_k=top_k, case_count=0, recall_at_k=0.0, mrr=0.0, hit_rate=0.0, per_case=results
        )
    return RetrievalSummary(
        top_k=top_k,
        case_count=len(scored),
        recall_at_k=mean(result.recall_at_k for result in scored),
        mrr=mean(result.reciprocal_rank for result in scored),
        hit_rate=mean(1.0 if result.hit else 0.0 for result in scored),
        per_case=results,
    )
