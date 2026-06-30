"""Citation accuracy metrics: cited pages / sections vs ground truth.

For each QA case, the answer's citations are compared with the expected pages
and section markers:

* ``page_match`` — fraction of expected pages covered by any citation's page
  range (this is the gated citation metric).
* ``section_match`` — fraction of expected section substrings found in any
  cited ``section_path``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean

from legal_ai.evals.datasets import QACase
from legal_ai.inference.qa_chain import Citation


@dataclass(frozen=True)
class CitationCaseResult:
    """Per-case citation scores."""

    case_id: str
    page_match: float
    section_match: float
    expected_pages: int
    expected_sections: int


@dataclass(frozen=True)
class CitationSummary:
    """Aggregated citation scores over cases that declare expectations."""

    case_count: int
    page_match: float
    section_match: float
    per_case: list[CitationCaseResult] = field(default_factory=list)


def _covered_pages(citations: Sequence[Citation]) -> set[int]:
    pages: set[int] = set()
    for citation in citations:
        pages.update(range(citation.page_start, citation.page_end + 1))
    return pages


def evaluate_citation_case(case: QACase, citations: Sequence[Citation]) -> CitationCaseResult:
    """Score citations of one answer against expected pages/sections."""

    expected_pages = set(case.expected_citation_pages)
    cited_pages = _covered_pages(citations)
    page_match = len(expected_pages & cited_pages) / len(expected_pages) if expected_pages else 0.0

    expected_sections = [section.lower() for section in case.expected_citation_sections]
    cited_sections = " ".join(citation.section_path.lower() for citation in citations)
    if expected_sections:
        matched = sum(1 for section in expected_sections if section in cited_sections)
        section_match = matched / len(expected_sections)
    else:
        section_match = 0.0

    return CitationCaseResult(
        case_id=case.id,
        page_match=page_match,
        section_match=section_match,
        expected_pages=len(expected_pages),
        expected_sections=len(expected_sections),
    )


def summarize_citations(results: list[CitationCaseResult]) -> CitationSummary:
    """Aggregate citation results per metric, ignoring undeclared expectations."""

    page_scored = [result for result in results if result.expected_pages > 0]
    section_scored = [result for result in results if result.expected_sections > 0]
    return CitationSummary(
        case_count=len(results),
        page_match=mean(result.page_match for result in page_scored) if page_scored else 0.0,
        section_match=(
            mean(result.section_match for result in section_scored) if section_scored else 0.0
        ),
        per_case=results,
    )
