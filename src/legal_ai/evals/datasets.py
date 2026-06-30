"""Golden evaluation datasets and JSONL loaders.

Two suites are supported:

* **retrieval** (`retrieval_cases.jsonl`) — query + ground-truth relevance. No
  LLM required, so it is the suite used as a CI gate.
* **qa** (`qa_golden.jsonl`) — question + expected answer keywords + expected
  citations. Requires the LLM (Ollama) and is meant for local / GPU runs.

Ground truth for retrieval is taken from the first populated signal, in
priority order: ``relevant_chunk_ids`` -> ``relevant_pages`` ->
``relevant_section_substrings``. Pages/sections keep cases authorable without
having to know the chunker's generated chunk ids.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class RetrievalCase(BaseModel):
    """One retrieval ground-truth case."""

    id: str
    query: str
    document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_pages: list[int] = Field(default_factory=list)
    relevant_section_substrings: list[str] = Field(default_factory=list)


class QACase(BaseModel):
    """One question-answering ground-truth case."""

    id: str
    question: str
    document_ids: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_citation_pages: list[int] = Field(default_factory=list)
    expected_citation_sections: list[str] = Field(default_factory=list)
    language: str = "en"


def _iter_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""

    rows: list[dict] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    """Load and validate retrieval cases from a JSONL file."""

    return [RetrievalCase.model_validate(row) for row in _iter_jsonl(Path(path))]


def load_qa_cases(path: str | Path) -> list[QACase]:
    """Load and validate QA cases from a JSONL file."""

    return [QACase.model_validate(row) for row in _iter_jsonl(Path(path))]
