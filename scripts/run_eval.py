"""CLI for running RAG evaluation suites and gating on metric thresholds.

Examples::

    python scripts/run_eval.py --suite retrieval --top-k 8 --output reports/eval.json
    python scripts/run_eval.py --suite qa --top-k 8 --output reports/eval_qa.json

The ``retrieval`` suite needs only Qdrant + the embedder (suitable for CI). The
``qa`` suite additionally needs Ollama. The process exits non-zero when a gated
metric falls below its threshold, unless ``--no-gate`` is passed.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

from legal_ai.config.logging import configure_logging, get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.evals import datasets
from legal_ai.evals.report import (
    EvalReport,
    Thresholds,
    build_gate,
    build_metadata,
    write_report,
)
from legal_ai.evals.runner import EvalRunner

_logger = get_logger("evals.cli")

_DEFAULT_CONTRACTS_DIR = "data/evals/contracts"
_DEFAULT_QA_FILE = "data/evals/qa_golden.jsonl"
_DEFAULT_RETRIEVAL_FILE = "data/evals/retrieval_cases.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation suites.")
    parser.add_argument("--suite", choices=["retrieval", "qa"], required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default="reports/eval.json")
    parser.add_argument("--contracts-dir", default=_DEFAULT_CONTRACTS_DIR)
    parser.add_argument("--qa-file", default=_DEFAULT_QA_FILE)
    parser.add_argument("--retrieval-file", default=_DEFAULT_RETRIEVAL_FILE)
    parser.add_argument("--collection", default=None, help="Override Qdrant collection.")
    parser.add_argument("--skip-ingest", action="store_true", help="Use already-indexed data.")
    parser.add_argument("--no-gate", action="store_true", help="Never fail on thresholds.")
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--min-citation-page", type=float, default=None)
    parser.add_argument("--min-answer-contains", type=float, default=None)
    return parser.parse_args(argv)


def _resolve_threshold(cli_value: float | None, env_name: str, default: float) -> float:
    if cli_value is not None:
        return cli_value
    env_value = os.getenv(env_name)
    return float(env_value) if env_value is not None else default


def resolve_thresholds(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        recall_at_k=_resolve_threshold(args.min_recall, "EVAL_MIN_RECALL", 0.70),
        citation_page_match=_resolve_threshold(
            args.min_citation_page, "EVAL_MIN_CITATION_PAGE", 0.60
        ),
        answer_contains=_resolve_threshold(
            args.min_answer_contains, "EVAL_MIN_ANSWER_CONTAINS", 0.65
        ),
    )


def resolve_settings(collection: str | None) -> Settings:
    settings = get_settings()
    if collection:
        return settings.model_copy(update={"qdrant_collection": collection})
    return settings


def build_retrieval_metrics(runner: EvalRunner, cases: list, top_k: int) -> dict[str, Any]:
    summary = runner.run_retrieval(cases, top_k)
    return {"retrieval": dataclasses.asdict(summary)}


def build_qa_metrics(runner: EvalRunner, cases: list, top_k: int) -> dict[str, Any]:
    answers, citations = runner.run_qa(cases, top_k)
    return {
        "answer": dataclasses.asdict(answers),
        "citation": dataclasses.asdict(citations),
    }


def _print_summary(report: EvalReport) -> None:
    status = "PASSED" if report.passed else "FAILED"
    _logger.info(f"Eval suite '{report.suite}' {status}")
    for check in report.checks:
        mark = "PASS" if check.passed else "FAIL"
        _logger.info(f"  [{mark}] {check.name}={check.value:.3f} (threshold {check.threshold:.2f})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level="INFO", log_format="text")

    cases_path = Path(args.retrieval_file if args.suite == "retrieval" else args.qa_file)
    if not cases_path.is_file():
        _logger.warning(f"No golden file at {cases_path}; nothing to evaluate (skipping gate).")
        return 0

    if args.suite == "retrieval":
        cases = datasets.load_retrieval_cases(cases_path)
    else:
        cases = datasets.load_qa_cases(cases_path)
    if not cases:
        _logger.warning(f"Golden file {cases_path} is empty; skipping gate.")
        return 0

    settings = resolve_settings(args.collection)
    top_k = args.top_k or settings.retrieval_top_k
    runner = EvalRunner(settings)

    contracts_indexed = 0
    if not args.skip_ingest:
        results = runner.ingest_contracts(args.contracts_dir)
        contracts_indexed = len(results)

    if args.suite == "retrieval":
        metrics = build_retrieval_metrics(runner, cases, top_k)
    else:
        metrics = build_qa_metrics(runner, cases, top_k)

    thresholds = resolve_thresholds(args)
    report = EvalReport(
        suite=args.suite,
        top_k=top_k,
        metadata=build_metadata(settings, args.suite, contracts_indexed),
        metrics=metrics,
        checks=build_gate(args.suite, metrics, thresholds),
    )
    json_path = write_report(report, args.output)
    _print_summary(report)
    _logger.info(f"Report written to {json_path} (+ {json_path.with_suffix('.md').name})")

    if args.no_gate or report.passed:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
