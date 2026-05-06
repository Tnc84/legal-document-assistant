"""Merge a trained LoRA adapter back into the base model and export to GGUF.

Usage:
    uv run python -m legal_ai.fine_tuning.merge_model \
        --base-model mistralai/Ministral-3B-Instruct \
        --adapter models/ministral-3b-cuad-lora \
        --output models/ministral-3b-cuad-merged

GGUF export depends on the external `llama.cpp` `convert_hf_to_gguf.py` script.
This module orchestrates the merge step and prints the suggested conversion
command when `--gguf-script` is provided.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from legal_ai.config.logging import configure_logging, get_logger

_logger = get_logger("fine_tuning.merge")


@dataclass(frozen=True)
class MergeArgs:
    base_model: str
    adapter_dir: Path
    output_dir: Path
    gguf_script: Path | None
    gguf_quantization: str


def parse_arguments() -> MergeArgs:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and optionally export GGUF")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gguf-script",
        type=Path,
        default=None,
        help="Path to llama.cpp convert_hf_to_gguf.py (optional)",
    )
    parser.add_argument("--gguf-quantization", default="q4_K_M")
    args = parser.parse_args()
    return MergeArgs(
        base_model=args.base_model,
        adapter_dir=args.adapter,
        output_dir=args.output,
        gguf_script=args.gguf_script,
        gguf_quantization=args.gguf_quantization,
    )


def merge_adapter(args: MergeArgs) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _logger.info(f"Loading base model {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    _logger.info(f"Applying adapter {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    merged = model.merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output_dir))
    _logger.info(f"Merged model saved to {args.output_dir}")


def maybe_export_gguf(args: MergeArgs) -> None:
    if args.gguf_script is None:
        _logger.info("Skipping GGUF export (no --gguf-script provided)")
        return
    if not args.gguf_script.is_file():
        raise FileNotFoundError(f"GGUF script not found: {args.gguf_script}")
    output_file = args.output_dir / f"model-{args.gguf_quantization}.gguf"
    command = [
        "python",
        str(args.gguf_script),
        str(args.output_dir),
        "--outfile",
        str(output_file),
        "--outtype",
        args.gguf_quantization,
    ]
    _logger.info(f"Running GGUF export: {' '.join(command)}")
    subprocess.run(command, check=True)
    _logger.info(f"GGUF written to {output_file}")


def main() -> None:
    configure_logging("INFO")
    args = parse_arguments()
    merge_adapter(args)
    maybe_export_gguf(args)


if __name__ == "__main__":
    main()
