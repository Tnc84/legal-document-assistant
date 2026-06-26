"""LoRA/QLoRA fine-tuning script for Ministral 3B on prepared CUAD JSONL.

Usage:
    uv pip install -e ".[finetune]"
    uv run python -m legal_ai.fine_tuning.train_qlora \
        --dataset data/processed/cuad_sft.jsonl \
        --base-model mistralai/Ministral-3B-Instruct \
        --output-dir models/ministral-3b-cuad-lora

Supports NVIDIA (bnb 4-bit) and AMD ROCm (fp16/bf16 LoRA fallback).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from legal_ai.config.logging import configure_logging, get_logger

_logger = get_logger("fine_tuning.train_qlora")


@dataclass
class TrainingArgs:
    dataset_path: Path
    base_model: str
    output_dir: Path
    num_epochs: float
    batch_size: int
    grad_accum: int
    learning_rate: float
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    quantization_mode: str


def parse_arguments() -> TrainingArgs:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for legal risk classifier")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared JSONL")
    parser.add_argument("--base-model", type=str, default="mistralai/Ministral-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantization-mode",
        choices=["auto", "bnb4", "fp16"],
        default="auto",
        help="auto=use bnb4 on NVIDIA, fp16 on AMD/CPU",
    )
    args = parser.parse_args()
    return TrainingArgs(
        dataset_path=args.dataset,
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        quantization_mode=args.quantization_mode,
    )


def format_record(record: dict) -> str:
    """Format a CUAD-prepared record into Mistral-style chat training text."""

    instruction = record.get("instruction", "").strip()
    user_input = record.get("input", "").strip()
    output = record.get("output", "").strip()
    return f"<s>[INST] {instruction}\n\n{user_input} [/INST] {output} </s>"


def run_training(args: TrainingArgs) -> None:
    """Run QLoRA SFT training. Heavy ML imports are local so the CLI module is cheap."""

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    _logger.info(f"Loading dataset {args.dataset_path}")
    dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    dataset = dataset.map(lambda row: {"text": format_record(row)})

    _logger.info(f"Loading base model {args.base_model}")
    use_cuda = torch.cuda.is_available()
    is_rocm = bool(getattr(torch.version, "hip", None))
    is_nvidia = use_cuda and not is_rocm
    effective_mode = _resolve_quantization_mode(args.quantization_mode, is_nvidia)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_supported = bool(
        use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )

    if effective_mode == "bnb4":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        bf16_enabled = True
        fp16_enabled = False
        _logger.info("Using bitsandbytes 4-bit quantization (NVIDIA path)")
    else:
        torch_dtype = (
            torch.bfloat16 if bf16_supported else (torch.float16 if use_cuda else torch.float32)
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
        bf16_enabled = bf16_supported
        fp16_enabled = bool(use_cuda and not bf16_supported)
        _logger.info("Using standard LoRA without bitsandbytes (AMD/CPU path)")

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_len,
        packing=False,
        seed=args.seed,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    _logger.info(f"Saved LoRA adapter to {args.output_dir}")


def _resolve_quantization_mode(requested_mode: str, is_nvidia: bool) -> str:
    if requested_mode == "auto":
        return "bnb4" if is_nvidia else "fp16"
    if requested_mode == "bnb4" and not is_nvidia:
        raise ValueError("quantization-mode=bnb4 is only supported on NVIDIA CUDA")
    return requested_mode


def main() -> None:
    configure_logging("INFO")
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_training(args)


if __name__ == "__main__":
    main()
