"""Evaluation report assembly: thresholds gate, JSON + markdown output.

The report bundles run metadata (for reproducibility), the computed metrics and
a list of gate checks. ``EvalReport.passed`` is ``True`` only when every gate
check passes, which the CLI maps to its process exit code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legal_ai.config.settings import Settings


@dataclass(frozen=True)
class Thresholds:
    """Minimum acceptable values for the gated metrics."""

    recall_at_k: float = 0.70
    citation_page_match: float = 0.60
    answer_contains: float = 0.65


@dataclass(frozen=True)
class GateCheck:
    """Result of comparing one metric against its threshold."""

    name: str
    value: float
    threshold: float
    passed: bool


@dataclass
class EvalReport:
    """Full evaluation report for one suite run."""

    suite: str
    top_k: int
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "top_k": self.top_k,
            "passed": self.passed,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "checks": [
                {
                    "name": check.name,
                    "value": round(check.value, 4),
                    "threshold": check.threshold,
                    "passed": check.passed,
                }
                for check in self.checks
            ],
        }

    def to_markdown(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"# RAG eval — suite `{self.suite}` — {status}",
            "",
            f"- top_k: {self.top_k}",
            f"- timestamp: {self.metadata.get('timestamp')}",
            f"- embedding_model: {self.metadata.get('embedding_model')}",
            f"- llm_model: {self.metadata.get('llm_model')}",
            f"- contracts_indexed: {self.metadata.get('contracts_indexed')}",
            "",
            "## Gate checks",
            "",
            "| Metric | Value | Threshold | Status |",
            "|---|---|---|---|",
        ]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"| {check.name} | {check.value:.3f} | {check.threshold:.2f} | {mark} |")
        if not self.checks:
            lines.append("| _none_ | - | - | - |")
        return "\n".join(lines) + "\n"


def build_metadata(settings: Settings, suite: str, contracts_indexed: int) -> dict[str, Any]:
    """Capture a reproducibility snapshot of the run configuration."""

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "suite": suite,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "llm_model": settings.ollama_model,
        "qdrant_collection": settings.qdrant_collection,
        "retrieval_dense_weight": settings.retrieval_dense_weight,
        "retrieval_bm25_weight": settings.retrieval_bm25_weight,
        "contracts_indexed": contracts_indexed,
    }


def build_gate(suite: str, metrics: dict[str, Any], thresholds: Thresholds) -> list[GateCheck]:
    """Build the gate checks relevant to the given suite."""

    if suite == "retrieval":
        recall = float(metrics["retrieval"]["recall_at_k"])
        return [
            GateCheck(
                name="recall_at_k",
                value=recall,
                threshold=thresholds.recall_at_k,
                passed=recall >= thresholds.recall_at_k,
            )
        ]
    if suite == "qa":
        contains = float(metrics["answer"]["contains_rate"])
        page_match = float(metrics["citation"]["page_match"])
        return [
            GateCheck(
                name="answer_contains",
                value=contains,
                threshold=thresholds.answer_contains,
                passed=contains >= thresholds.answer_contains,
            ),
            GateCheck(
                name="citation_page_match",
                value=page_match,
                threshold=thresholds.citation_page_match,
                passed=page_match >= thresholds.citation_page_match,
            ),
        ]
    raise ValueError(f"Unknown suite: {suite}")


def write_report(report: EvalReport, output_path: str | Path) -> Path:
    """Write the JSON report plus a sibling markdown summary; return JSON path."""

    json_path = Path(output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), "utf-8")
    json_path.with_suffix(".md").write_text(report.to_markdown(), "utf-8")
    return json_path
