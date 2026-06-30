"""Answer quality metrics: keyword coverage and lexical overlap (RO + EN).

Text is normalized by lowercasing and stripping diacritics (NFKD) so Romanian
forms such as "reziliere" / "rezilière" match regardless of accents. Two
signals are produced per case:

* ``contains_rate`` — fraction of ``expected_answer_contains`` keywords present
  in the answer (substring match). This is the gated answer metric.
* ``lexical_overlap`` — fraction of expected keyword *tokens* present in the
  answer tokens (a softer recall-style signal).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from statistics import mean

from legal_ai.evals.datasets import QACase

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class AnswerCaseResult:
    """Per-case answer scores."""

    case_id: str
    contains_rate: float
    lexical_overlap: float
    expected_total: int


@dataclass(frozen=True)
class AnswerSummary:
    """Aggregated answer scores over cases that declare expected keywords."""

    case_count: int
    contains_rate: float
    lexical_overlap: float
    per_case: list[AnswerCaseResult] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Lowercase and remove combining diacritics for accent-insensitive match."""

    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokenize(text: str) -> set[str]:
    """Tokenize into a set of normalized word tokens."""

    return set(_TOKEN_RE.findall(normalize_text(text)))


def evaluate_answer_case(case: QACase, answer: str) -> AnswerCaseResult:
    """Score one generated answer against expected keywords."""

    expected = case.expected_answer_contains
    normalized_answer = normalize_text(answer)
    if expected:
        present = sum(1 for keyword in expected if normalize_text(keyword) in normalized_answer)
        contains_rate = present / len(expected)
        expected_tokens = set().union(*(tokenize(keyword) for keyword in expected))
    else:
        contains_rate = 0.0
        expected_tokens = set()

    answer_tokens = tokenize(answer)
    lexical_overlap = (
        len(answer_tokens & expected_tokens) / len(expected_tokens) if expected_tokens else 0.0
    )
    return AnswerCaseResult(
        case_id=case.id,
        contains_rate=contains_rate,
        lexical_overlap=lexical_overlap,
        expected_total=len(expected),
    )


def summarize_answers(results: list[AnswerCaseResult]) -> AnswerSummary:
    """Aggregate answer results, ignoring cases without expected keywords."""

    scored = [result for result in results if result.expected_total > 0]
    if not scored:
        return AnswerSummary(case_count=0, contains_rate=0.0, lexical_overlap=0.0, per_case=results)
    return AnswerSummary(
        case_count=len(scored),
        contains_rate=mean(result.contains_rate for result in scored),
        lexical_overlap=mean(result.lexical_overlap for result in scored),
        per_case=results,
    )
