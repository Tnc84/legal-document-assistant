"""Risk clause detector: rule-based pre-filter + LLM-backed structured classifier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from pydantic import BaseModel, Field, ValidationError

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.inference.llm_client import LLMClient, load_prompt, safe_json_loads
from legal_ai.retrieval.vector_store import QdrantVectorStore, SearchHit

_logger = get_logger("inference.risk")

RiskCategory = Literal[
    "penalty",
    "liability_cap",
    "exclusivity",
    "auto_renewal",
    "unilateral_termination",
    "unfavorable_jurisdiction",
    "data_protection",
    "ip_assignment",
    "confidentiality",
    "change_control",
]
Severity = Literal["low", "medium", "high"]


_RULE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "penalty": [
        re.compile(r"\bpenalt(?:y|ies|ate?)\b", re.IGNORECASE),
        re.compile(r"\bliquidated damages\b", re.IGNORECASE),
        re.compile(r"\bdaune\s+(?:cominatorii|interese)\b", re.IGNORECASE),
    ],
    "liability_cap": [
        re.compile(r"\blimit(?:ation)?\s+of\s+liability\b", re.IGNORECASE),
        re.compile(r"\blimitarea\s+r[ăa]spunderii\b", re.IGNORECASE),
        re.compile(r"\bindemnif", re.IGNORECASE),
    ],
    "exclusivity": [
        re.compile(r"\bexclusiv", re.IGNORECASE),
        re.compile(r"\bnon[-\s]?compete\b", re.IGNORECASE),
        re.compile(r"\bneconcuren", re.IGNORECASE),
    ],
    "auto_renewal": [
        re.compile(r"\bauto(?:matic)?\s+renew", re.IGNORECASE),
        re.compile(r"\bre[îi]nnoir", re.IGNORECASE),
        re.compile(r"\btacit\s+(?:re|prelung)", re.IGNORECASE),
    ],
    "unilateral_termination": [
        re.compile(r"\bunilateral", re.IGNORECASE),
        re.compile(r"\btermin(?:ate|ation)\b", re.IGNORECASE),
        re.compile(r"\breziliere\b", re.IGNORECASE),
    ],
    "unfavorable_jurisdiction": [
        re.compile(r"\bgoverning\s+law\b", re.IGNORECASE),
        re.compile(r"\bjurisdic", re.IGNORECASE),
        re.compile(r"\blegea\s+aplicabil", re.IGNORECASE),
    ],
    "data_protection": [
        re.compile(r"\bGDPR\b"),
        re.compile(r"\bdate\s+(?:cu\s+)?caracter\s+personal\b", re.IGNORECASE),
        re.compile(r"\bdata\s+(?:processing|protection)\b", re.IGNORECASE),
    ],
    "ip_assignment": [
        re.compile(r"\bintellectual\s+property\b", re.IGNORECASE),
        re.compile(r"\bproprietate\s+intelectual", re.IGNORECASE),
        re.compile(r"\bassign(?:ment)?\s+of\s+rights\b", re.IGNORECASE),
    ],
    "confidentiality": [
        re.compile(r"\bconfidential", re.IGNORECASE),
        re.compile(r"\bnon[-\s]?disclosure\b", re.IGNORECASE),
    ],
    "change_control": [
        re.compile(r"\bchange\s+of\s+(?:price|scope|control)\b", re.IGNORECASE),
        re.compile(r"\bmodificare\s+unilateral", re.IGNORECASE),
    ],
}


class RiskFinding(BaseModel):
    """Validated finding produced by the LLM."""

    category: RiskCategory
    severity: Severity
    source_text: str = Field(min_length=1)
    page: int = Field(ge=0)
    section: str = ""
    rationale: str = Field(min_length=1)
    recommendation: str = ""


class RiskFindings(BaseModel):
    findings: list[RiskFinding] = Field(default_factory=list)


@dataclass(frozen=True)
class RiskReport:
    """Aggregated risk findings for a document."""

    document_id: str
    findings: list[RiskFinding]


class RiskDetector:
    """Combine rule-based candidate selection with LLM JSON-mode classification."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._settings = settings or get_settings()
        self._system_prompt = load_prompt("risk_system")

    def analyze_document(self, document_id: str, max_chunks: int = 200) -> RiskReport:
        chunks = self._vector_store.fetch_document_chunks(document_id, limit=max_chunks)
        if not chunks:
            return RiskReport(document_id=document_id, findings=[])

        candidates = self._select_candidates(chunks)
        if not candidates:
            return RiskReport(document_id=document_id, findings=[])

        findings: list[RiskFinding] = []
        for chunk in candidates:
            findings.extend(self._classify_chunk(chunk))
        deduped = self._deduplicate(findings)
        return RiskReport(document_id=document_id, findings=deduped)

    def _select_candidates(self, chunks: list[SearchHit]) -> list[SearchHit]:
        selected: list[SearchHit] = []
        for chunk in chunks:
            if any(
                pattern.search(chunk.text)
                for patterns in _RULE_PATTERNS.values()
                for pattern in patterns
            ):
                selected.append(chunk)
        _logger.info(
            f"Risk pre-filter selected {len(selected)}/{len(chunks)} chunks for LLM classification"
        )
        return selected

    def _classify_chunk(self, chunk: SearchHit) -> Iterable[RiskFinding]:
        user_prompt = self._build_user_prompt(chunk)
        try:
            raw = self._llm.complete(self._system_prompt, user_prompt, json_mode=True)
            payload = safe_json_loads(raw)
            parsed = RiskFindings.model_validate(payload)
            return parsed.findings
        except (ValidationError, ValueError) as exc:
            _logger.warning(f"Skipping chunk {chunk.chunk_id} due to parse error: {exc}")
            return []

    @staticmethod
    def _build_user_prompt(chunk: SearchHit) -> str:
        return (
            f"Contract: {chunk.document_title}\n"
            f"Page: {chunk.page_start}\n"
            f"Section: {chunk.section_path or 'N/A'}\n\n"
            f"Excerpt:\n{chunk.text}\n\n"
            "Return the JSON described in the system prompt."
        )

    @staticmethod
    def _deduplicate(findings: list[RiskFinding]) -> list[RiskFinding]:
        seen: set[tuple[str, str, int]] = set()
        unique: list[RiskFinding] = []
        for finding in findings:
            key = (finding.category, finding.source_text[:200].strip().lower(), finding.page)
            if key in seen:
                continue
            seen.add(key)
            unique.append(finding)
        unique.sort(key=lambda f: (_severity_rank(f.severity), f.page))
        return unique


def _severity_rank(severity: Severity) -> int:
    return {"high": 0, "medium": 1, "low": 2}[severity]
