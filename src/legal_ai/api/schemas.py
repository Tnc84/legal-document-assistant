"""Pydantic request/response schemas for the FastAPI layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    title: str
    page_count: int
    chunk_count: int


class QARequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=30)


class CitationModel(BaseModel):
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section_path: str
    snippet: str
    score: float


class QAResponseModel(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]


class RiskRequest(BaseModel):
    document_id: str = Field(min_length=1)
    max_chunks: int = Field(default=200, ge=1, le=2000)


class RiskFindingModel(BaseModel):
    category: str
    severity: Literal["low", "medium", "high"]
    source_text: str
    page: int
    section: str
    rationale: str
    recommendation: str


class RiskResponseModel(BaseModel):
    document_id: str
    findings: list[RiskFindingModel]


class ClauseDiffModel(BaseModel):
    change_type: Literal["added", "removed", "modified", "unchanged"]
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


class ComparisonResponseModel(BaseModel):
    left_title: str
    right_title: str
    total_left: int
    total_right: int
    total_risk_delta: int
    diffs: list[ClauseDiffModel]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant: bool
    ollama: bool
    embedding_model: str
    llm_model: str
