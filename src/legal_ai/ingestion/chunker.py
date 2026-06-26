"""Semantic clause-aware chunker for contracts."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from legal_ai.ingestion.parser import ParsedDocument
from legal_ai.utils.text import estimate_tokens

_CLAUSE_HEADER_RE = re.compile(
    r"""
    ^\s*
    (?:
        (?:Art(?:icle|icol)\.?|Cl(?:ause|auza)\.?|Section|Sec(?:tiunea)?\.?)\s+
        [\dIVXLC]+(?:\.\d+)*
        |
        \d+(?:\.\d+){0,3}\s*[\.\)]
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class Chunk:
    """A retrievable chunk with structural metadata for citation."""

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page_start: int
    page_end: int
    section_path: str
    token_estimate: int
    metadata: dict[str, str] = field(default_factory=dict)


class SemanticChunker:
    """Chunk a parsed document on clause boundaries with token budget."""

    def __init__(self, max_tokens: int, overlap_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, document: ParsedDocument, document_id: str) -> list[Chunk]:
        segments = list(self._iter_clause_segments(document))
        merged = self._merge_segments(segments)
        chunks: list[Chunk] = []
        for index, segment in enumerate(merged):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}::chunk_{index:05d}",
                    document_id=document_id,
                    document_title=document.title,
                    text=segment.text,
                    page_start=segment.page_start,
                    page_end=segment.page_end,
                    section_path=segment.section_path or "",
                    token_estimate=estimate_tokens(segment.text),
                    metadata={
                        "source_path": document.source_path,
                        "page_start": str(segment.page_start),
                        "page_end": str(segment.page_end),
                        "section_path": segment.section_path or "",
                    },
                )
            )
        return chunks

    def _iter_clause_segments(self, document: ParsedDocument) -> Iterable[_Segment]:
        current_section = ""
        for page in document.pages:
            if not page.text:
                continue
            buffer: list[str] = []
            buffer_section = current_section
            for raw_line in page.text.split("\n"):
                line = raw_line.strip()
                if not line:
                    if buffer:
                        buffer.append("")
                    continue
                if _CLAUSE_HEADER_RE.match(line):
                    if buffer:
                        yield _Segment(
                            text="\n".join(buffer).strip(),
                            page_start=page.page_number,
                            page_end=page.page_number,
                            section_path=buffer_section,
                        )
                        buffer = []
                    current_section = line[:200]
                    buffer_section = current_section
                buffer.append(line)
            if buffer:
                yield _Segment(
                    text="\n".join(buffer).strip(),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section_path=buffer_section,
                )

    def _merge_segments(self, segments: list[_Segment]) -> list[_Segment]:
        merged: list[_Segment] = []
        for segment in segments:
            if not segment.text:
                continue
            if not merged:
                merged.append(segment)
                continue
            last = merged[-1]
            combined_tokens = estimate_tokens(last.text) + estimate_tokens(segment.text)
            if combined_tokens <= self._max_tokens:
                merged[-1] = _Segment(
                    text=f"{last.text}\n\n{segment.text}",
                    page_start=last.page_start,
                    page_end=max(last.page_end, segment.page_end),
                    section_path=last.section_path or segment.section_path,
                )
            else:
                merged.append(segment)

        return self._apply_overlap(merged)

    def _apply_overlap(self, segments: list[_Segment]) -> list[_Segment]:
        if self._overlap_tokens == 0 or len(segments) < 2:
            return segments
        overlapped: list[_Segment] = [segments[0]]
        for previous, current in zip(segments, segments[1:], strict=False):
            tail = self._tail_within_tokens(previous.text, self._overlap_tokens)
            new_text = f"{tail}\n\n{current.text}" if tail else current.text
            overlapped.append(
                _Segment(
                    text=new_text,
                    page_start=previous.page_end if tail else current.page_start,
                    page_end=current.page_end,
                    section_path=current.section_path or previous.section_path,
                )
            )
        return overlapped

    @staticmethod
    def _tail_within_tokens(text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        budget = max_tokens * 4
        if len(text) <= budget:
            return text
        return text[-budget:]


@dataclass(frozen=True)
class _Segment:
    text: str
    page_start: int
    page_end: int
    section_path: str
