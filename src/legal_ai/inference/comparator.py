"""Semantic document comparator producing per-clause diff with risk delta."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.inference.llm_client import LLMClient, load_prompt, safe_json_loads
from legal_ai.ingestion.chunker import Chunk, SemanticChunker
from legal_ai.ingestion.embedder import Embedder
from legal_ai.ingestion.parser import PdfParser

_logger = get_logger("inference.comparator")

ChangeType = Literal["added", "removed", "modified", "unchanged"]


class ChangeAnalysis(BaseModel):
    change_type: ChangeType
    summary: str = Field(min_length=1)
    risk_delta: int = Field(ge=-3, le=3)
    rationale: str = ""


@dataclass(frozen=True)
class ClauseDiff:
    """One per-clause difference between two contract versions."""

    change_type: ChangeType
    similarity: float
    risk_delta: int
    summary: str
    rationale: str
    left_section: str | None
    right_section: str | None
    left_page: int | None
    right_page: int | None
    left_text: str | None
    right_text: str | None


@dataclass(frozen=True)
class ComparisonReport:
    """Aggregated diff across an entire pair of contracts."""

    left_title: str
    right_title: str
    total_left: int
    total_right: int
    diffs: list[ClauseDiff]

    @property
    def total_risk_delta(self) -> int:
        return sum(diff.risk_delta for diff in self.diffs)


class DocumentComparator:
    """Compare two contracts at clause granularity using embeddings + LLM analysis."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        similarity_threshold: float = 0.82,
        text_dup_threshold: int = 95,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or Embedder(self._settings)
        self._llm = llm
        self._parser = PdfParser(extract_tables=False)
        self._chunker = SemanticChunker(
            max_tokens=self._settings.chunk_size_tokens,
            overlap_tokens=0,
        )
        self._system_prompt = load_prompt("compare_system")
        self._similarity_threshold = similarity_threshold
        self._text_dup_threshold = text_dup_threshold

    def compare_pdfs(self, left_pdf: str | Path, right_pdf: str | Path) -> ComparisonReport:
        left_doc = self._parser.parse(left_pdf)
        right_doc = self._parser.parse(right_pdf)

        left_chunks = self._chunker.chunk(left_doc, document_id="left")
        right_chunks = self._chunker.chunk(right_doc, document_id="right")
        if not left_chunks and not right_chunks:
            return ComparisonReport(
                left_title=left_doc.title,
                right_title=right_doc.title,
                total_left=0,
                total_right=0,
                diffs=[],
            )

        left_vectors = self._embed(left_chunks)
        right_vectors = self._embed(right_chunks)

        matches, unmatched_left, unmatched_right = self._match(
            left_chunks, right_chunks, left_vectors, right_vectors
        )

        diffs: list[ClauseDiff] = []
        for left_idx, right_idx, similarity in matches:
            left = left_chunks[left_idx]
            right = right_chunks[right_idx]
            if fuzz.ratio(left.text, right.text) >= self._text_dup_threshold:
                diffs.append(self._unchanged_diff(left, right, similarity))
                continue
            analysis = self._analyze_with_llm(left.text, right.text, "modified")
            diffs.append(self._modified_diff(left, right, similarity, analysis))

        for idx in unmatched_left:
            left = left_chunks[idx]
            analysis = self._analyze_with_llm(left.text, None, "removed")
            diffs.append(self._side_only_diff(left, side="left", analysis=analysis))

        for idx in unmatched_right:
            right = right_chunks[idx]
            analysis = self._analyze_with_llm(None, right.text, "added")
            diffs.append(self._side_only_diff(right, side="right", analysis=analysis))

        diffs.sort(key=lambda d: (-abs(d.risk_delta), d.change_type))

        return ComparisonReport(
            left_title=left_doc.title,
            right_title=right_doc.title,
            total_left=len(left_chunks),
            total_right=len(right_chunks),
            diffs=diffs,
        )

    def _embed(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self._embedder.dimension), dtype=np.float32)
        vectors = self._embedder.encode_passages([chunk.text for chunk in chunks])
        return np.asarray(vectors, dtype=np.float32)

    def _match(
        self,
        left_chunks: list[Chunk],
        right_chunks: list[Chunk],
        left_vectors: np.ndarray,
        right_vectors: np.ndarray,
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        if not left_chunks or not right_chunks:
            return [], list(range(len(left_chunks))), list(range(len(right_chunks)))

        similarity = left_vectors @ right_vectors.T
        used_right: set[int] = set()
        matches: list[tuple[int, int, float]] = []
        unmatched_left: list[int] = []

        for left_idx in range(len(left_chunks)):
            row = similarity[left_idx]
            order = np.argsort(-row)
            picked: int | None = None
            for candidate in order:
                if candidate in used_right:
                    continue
                if float(row[candidate]) < self._similarity_threshold:
                    break
                picked = int(candidate)
                break
            if picked is None:
                unmatched_left.append(left_idx)
            else:
                used_right.add(picked)
                matches.append((left_idx, picked, float(row[picked])))

        unmatched_right = [idx for idx in range(len(right_chunks)) if idx not in used_right]
        return matches, unmatched_left, unmatched_right

    def _analyze_with_llm(
        self,
        left_text: str | None,
        right_text: str | None,
        hint: ChangeType,
    ) -> ChangeAnalysis:
        if self._llm is None:
            return self._heuristic_analysis(left_text, right_text, hint)

        user_prompt = (
            f"Suggested change_type hint: {hint}\n\n"
            f"LEFT clause:\n{left_text or '<missing>'}\n\n"
            f"RIGHT clause:\n{right_text or '<missing>'}\n\n"
            "Return the JSON described in the system prompt."
        )
        try:
            raw = self._llm.complete(self._system_prompt, user_prompt, json_mode=True)
            payload = safe_json_loads(raw)
            return ChangeAnalysis.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            _logger.warning(f"Comparator LLM parse fallback ({hint}): {exc}")
            return self._heuristic_analysis(left_text, right_text, hint)

    @staticmethod
    def _heuristic_analysis(
        left_text: str | None,
        right_text: str | None,
        hint: ChangeType,
    ) -> ChangeAnalysis:
        delta = {"added": 1, "removed": -1, "modified": 1, "unchanged": 0}[hint]
        if hint == "modified" and left_text and right_text:
            similarity = fuzz.token_set_ratio(left_text, right_text) / 100.0
            delta = 1 if similarity < 0.6 else 0
        return ChangeAnalysis(
            change_type=hint,
            summary=f"Auto-generated {hint} (LLM unavailable)",
            risk_delta=delta,
            rationale="Heuristic fallback without LLM analysis.",
        )

    @staticmethod
    def _unchanged_diff(left: Chunk, right: Chunk, similarity: float) -> ClauseDiff:
        return ClauseDiff(
            change_type="unchanged",
            similarity=similarity,
            risk_delta=0,
            summary="No substantive change detected.",
            rationale="High textual similarity above duplicate threshold.",
            left_section=left.section_path,
            right_section=right.section_path,
            left_page=left.page_start,
            right_page=right.page_start,
            left_text=left.text,
            right_text=right.text,
        )

    @staticmethod
    def _modified_diff(
        left: Chunk,
        right: Chunk,
        similarity: float,
        analysis: ChangeAnalysis,
    ) -> ClauseDiff:
        return ClauseDiff(
            change_type=analysis.change_type,
            similarity=similarity,
            risk_delta=analysis.risk_delta,
            summary=analysis.summary,
            rationale=analysis.rationale,
            left_section=left.section_path,
            right_section=right.section_path,
            left_page=left.page_start,
            right_page=right.page_start,
            left_text=left.text,
            right_text=right.text,
        )

    @staticmethod
    def _side_only_diff(
        chunk: Chunk,
        *,
        side: Literal["left", "right"],
        analysis: ChangeAnalysis,
    ) -> ClauseDiff:
        is_left = side == "left"
        return ClauseDiff(
            change_type=analysis.change_type,
            similarity=0.0,
            risk_delta=analysis.risk_delta,
            summary=analysis.summary,
            rationale=analysis.rationale,
            left_section=chunk.section_path if is_left else None,
            right_section=None if is_left else chunk.section_path,
            left_page=chunk.page_start if is_left else None,
            right_page=None if is_left else chunk.page_start,
            left_text=chunk.text if is_left else None,
            right_text=None if is_left else chunk.text,
        )
