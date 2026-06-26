"""Convert the CUAD dataset into instruction-tuning JSONL for QLoRA.

Usage:
    uv run python -m legal_ai.fine_tuning.prepare_cuad \
        --cuad-json data/cuad/CUAD_v1.json \
        --output data/processed/cuad_sft.jsonl

The CUAD JSON follows the SQuAD-like structure with `data -> paragraphs -> qas`.
Each `qas[i].question` ends with a category label, and `answers[i].text` contains
the verbatim clause excerpt. We keep only categories present in our taxonomy.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from legal_ai.config.logging import configure_logging, get_logger
from legal_ai.fine_tuning.taxonomy import map_cuad_label

_logger = get_logger("fine_tuning.prepare_cuad")

_INSTRUCTION = (
    "You are a legal risk classifier. Given a contract excerpt, return JSON "
    '{"category": <risk_category>, "severity": "low"|"medium"|"high", '
    '"source_text": <verbatim quote>, "rationale": <one sentence>}. '
    "Use only categories from the project taxonomy; if the excerpt has no risk, "
    'return {"category": "none", ...}.'
)


@dataclass(frozen=True)
class SftRecord:
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict[str, str]:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CUAD for SFT")
    parser.add_argument("--cuad-json", type=Path, required=True, help="Path to CUAD_v1.json")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--negative-ratio", type=float, default=0.5, help="Ratio of negatives")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-chars", type=int, default=2400)
    return parser.parse_args()


def load_cuad(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"CUAD file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_records(cuad: dict, max_context_chars: int) -> tuple[list[SftRecord], list[str]]:
    positives: list[SftRecord] = []
    contexts: list[str] = []
    for entry in cuad.get("data", []):
        for paragraph in entry.get("paragraphs", []):
            context = (paragraph.get("context") or "").strip()
            if not context:
                continue
            contexts.append(context[:max_context_chars])
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers") or []
                if qa.get("is_impossible") or not answers:
                    continue
                category = _category_from_question(question)
                if category is None:
                    continue
                for answer in answers:
                    snippet = (answer.get("text") or "").strip()
                    if not snippet:
                        continue
                    output = json.dumps(
                        {
                            "category": category,
                            "severity": "medium",
                            "source_text": snippet,
                            "rationale": (
                                f"Detected {category.replace('_', ' ')} clause via CUAD supervision."
                            ),
                        },
                        ensure_ascii=False,
                    )
                    positives.append(
                        SftRecord(
                            instruction=_INSTRUCTION,
                            input=context[:max_context_chars],
                            output=output,
                        )
                    )
    return positives, contexts


def build_negatives(
    contexts: list[str],
    count: int,
    seed: int,
    max_context_chars: int,
) -> list[SftRecord]:
    if count <= 0 or not contexts:
        return []
    rng = random.Random(seed)
    sampled = rng.sample(contexts, k=min(count, len(contexts)))
    negatives: list[SftRecord] = []
    for context in sampled:
        output = json.dumps(
            {
                "category": "none",
                "severity": "low",
                "source_text": "",
                "rationale": "No risk clause detected in this excerpt.",
            },
            ensure_ascii=False,
        )
        negatives.append(
            SftRecord(
                instruction=_INSTRUCTION,
                input=context[:max_context_chars],
                output=output,
            )
        )
    return negatives


def write_jsonl(records: list[SftRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _category_from_question(question: str) -> str | None:
    if "?" not in question:
        return None
    head, _, tail = question.rpartition("?")
    label = tail.strip() or head.split(":")[-1].strip()
    if not label:
        return None
    return map_cuad_label(label)


def main() -> None:
    configure_logging("INFO")
    args = parse_arguments()
    cuad = load_cuad(args.cuad_json)
    positives, contexts = build_records(cuad, args.max_context_chars)
    negative_count = int(len(positives) * args.negative_ratio)
    negatives = build_negatives(contexts, negative_count, args.seed, args.max_context_chars)
    all_records = positives + negatives
    rng = random.Random(args.seed)
    rng.shuffle(all_records)
    write_jsonl(all_records, args.output)
    _logger.info(
        f"Wrote {len(all_records)} records ({len(positives)} positives, "
        f"{len(negatives)} negatives) to {args.output}"
    )


if __name__ == "__main__":
    main()
