"""Training orchestration: load FTPO pairs, build dataset, train LoRA, merge.

Ties together the data pipeline (ftpo_data) and the loss (reference_ftpo_trainer)
into a single train-and-merge flow. Handles LoRA config, optional early-layer
freezing, early stopping on chosen_win, and adapter merge.

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

import getpass
import inspect
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from thoxa_antidoom.config import TrainConfig
from thoxa_antidoom.ftpo_data import (
    FTPODataset,
    has_trimmed_source_prompt,
    read_jsonl,
    row_source_label,
    sample_rows,
)
from thoxa_antidoom.reference_ftpo_trainer import FTPOTrainer, ThresholdStop

logger = logging.getLogger(__name__)


def _format_float(value: float) -> str:
    text = f"{value:.8g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        sign = ""
        if exponent.startswith(("+", "-")):
            sign, exponent = exponent[0], exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        if sign == "+":
            sign = ""
        text = f"{mantissa}e{sign}{exponent}"
    return text


def _format_epochs(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else _format_float(value)


def _format_counter(counter: Counter[str], limit: int = 8) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def _sanitize_name_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip())
    return value.strip("-_") or "unknown"


def _model_basename(model_name: str) -> str:
    return model_name.rstrip("/").rsplit("/", 1)[-1]


def _parse_parent_model_name(model_name: str) -> dict[str, str]:
    name = _model_basename(model_name)
    size_match = re.search(r"(\d+B)", name, flags=re.IGNORECASE)
    parent_match = re.search(r"_(\d+)_HF(?:_|$)", name)
    if parent_match is None:
        numbers = re.findall(r"\d+", name)
        parent_id = numbers[-1] if numbers else "unknown"
    else:
        parent_id = parent_match.group(1)
    return {
        "size": size_match.group(1) if size_match else "model",
        "parent_id": parent_id,
    }


def build_merged_model_name(cfg: TrainConfig, *, train_examples: int) -> str:
    """Generate a descriptive HuggingFace-style name for the merged model."""
    parent = _parse_parent_model_name(cfg.model_name)
    user = _sanitize_name_part(getpass.getuser())
    job_id = _sanitize_name_part(os.environ.get("SLURM_JOB_ID") or "local")
    parts = [
        f"{user}_antidoom{parent['size']}",
        f"from{parent['parent_id']}",
        f"{train_examples}-samples",
        f"lr{_format_float(cfg.learning_rate)}",
        f"ep{_format_epochs(cfg.num_epochs)}",
        job_id,
        "HF",
    ]
    return _sanitize_name_part("_".join(parts))


def resolve_merged_output_dir(cfg: TrainConfig, *, train_examples: int) -> Path:
    if not cfg.auto_merged_output_name:
        return cfg.merged_output_dir
    return cfg.merged_output_dir.parent / build_merged_model_name(
        cfg, train_examples=train_examples
    )


def filter_trimmed_source_prompt_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Remove rows whose source prompt was trimmed during generation."""
    kept = [row for row in rows if not has_trimmed_source_prompt(row)]
    return kept, len(rows) - len(kept)


def _make_dpo_config(cfg: TrainConfig) -> DPOConfig:
    kwargs = {
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "warmup_ratio": cfg.warmup_ratio,
        "num_train_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "logging_steps": 5,
        "optim": cfg.optim,
        "seed": cfg.seed,
        "output_dir": str(cfg.output_dir),
        "max_length": cfg.max_seq_length,
        "max_prompt_length": cfg.max_seq_length // 2,
        "beta": 0.1,
        "weight_decay": cfg.weight_decay,
        "report_to": "tensorboard",
        "lr_scheduler_type": "linear",
        "bf16": cfg.bf16,
        "fp16": False,
        "remove_unused_columns": False,
        "disable_tqdm": False,
        "max_grad_norm": 2.5,
    }
    supported_params = inspect.signature(DPOConfig).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in supported_params}
    dropped = sorted(set(kwargs) - set(supported_kwargs))
    if dropped:
        logger.info("DPOConfig does not support: %s; dropping", ", ".join(dropped))
    return DPOConfig(**supported_kwargs)


def _make_lora_config(cfg: TrainConfig) -> LoraConfig:
    kwargs = {
        "r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "bias": "none",
        "target_modules": cfg.target_modules,
    }
    if cfg.lora_ensure_weight_tying:
        kwargs["modules_to_save"] = ["lm_head", "embed_tokens"]
    return LoraConfig(**kwargs)


def _freeze_lora_to_last_k_layers(model, k: int, target_modules: tuple[str, ...]) -> None:
    """Disable LoRA adapters on all but the last ``k`` transformer layers."""
    frozen = 0
    trainable = 0
    layer_pattern = re.compile(r"\.(\d+)\.")

    trainable_layer_nums = set()
    all_layer_nums = set()
    for name, _ in model.named_modules():
        m = layer_pattern.search(name)
        if m:
            ln = int(m.group(1))
            all_layer_nums.add(ln)
            if any(tm in name for tm in target_modules):
                pass

    if all_layer_nums:
        max_layer = max(all_layer_nums)
        trainable_layer_nums = set(range(max_layer - k + 1, max_layer + 1))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        m = layer_pattern.search(name)
        if m:
            ln = int(m.group(1))
            if ln not in trainable_layer_nums:
                param.requires_grad = False
                frozen += 1
            else:
                trainable += 1
    logger.info("froze %d LoRA params, kept %d trainable (last %d layers)", frozen, trainable, k)


def log_trainable_lora_by_layer(model) -> None:
    """Log how many trainable parameters exist per transformer layer."""
    layer_pattern = re.compile(r"\.(\d+)\.")
    by_layer: Counter[str] = Counter()
    for name, param in model.named_parameters():
        if param.requires_grad:
            m = layer_pattern.search(name)
            layer = f"layer_{m.group(1)}" if m else "other"
            by_layer[layer] += param.numel()
    logger.info("trainable params by layer: %s", _format_counter(by_layer, limit=20))


def _save_merged_model(trainer, tokenizer, cfg: TrainConfig) -> None:
    """Merge the LoRA adapter into the base model and save."""
    if not cfg.merge_lora:
        return
    merged_dir = cfg.merged_output_dir
    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("merging LoRA adapter into %s", merged_dir)

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    logger.info("saved merged model to %s", merged_dir)


def run_training(cfg: TrainConfig, pairs_path: Path) -> tuple[Path, Path | None]:
    """Run the FTPO training loop. Returns (adapter_dir, merged_dir_or_None)."""
    dataset_path = cfg.dataset_jsonl or pairs_path
    logger.info("loading FTPO pairs from %s", dataset_path)
    rows = read_jsonl(dataset_path)
    logger.info("loaded %d raw FTPO rows", len(rows))

    logger.info(
        "rows by rejected token: %s",
        _format_counter(Counter(r.get("rejected_decoded") for r in rows)),
    )
    logger.info(
        "raw rows by chosen candidate count: %s",
        _format_counter(Counter(str(len(r.get("multi_chosen_decoded") or [])) for r in rows)),
    )

    rows_without_trimmed, trimmed_count = filter_trimmed_source_prompt_rows(rows)
    if trimmed_count:
        logger.info(
            "%d/%d rows pass trimmed source_prompt filter; discarded %d",
            len(rows_without_trimmed), len(rows), trimmed_count,
        )

    min_chosen_rows = [
        row
        for row in rows_without_trimmed
        if len(row.get("multi_chosen_decoded") or []) >= cfg.min_chosen_tokens
    ]
    logger.info(
        "%d/%d rows pass min_chosen_tokens=%d before tokenization",
        len(min_chosen_rows), len(rows_without_trimmed), cfg.min_chosen_tokens,
    )

    sampled_rows = sample_rows(
        rows_without_trimmed,
        max_train_examples=cfg.max_train_examples,
        rejected_strength=cfg.rejected_regularisation_strength,
        min_chosen_tokens=cfg.min_chosen_tokens,
        seed=cfg.seed,
        source_balance_mode=cfg.source_balance_mode,
    )
    logger.info(
        "sampled %d/%d rows after pre-tokenization filters "
        "(max_train_examples=%s, min_chosen_tokens=%d, source_balance_mode=%s)",
        len(sampled_rows), len(rows_without_trimmed),
        cfg.max_train_examples, cfg.min_chosen_tokens, cfg.source_balance_mode,
    )
    logger.info(
        "sampled rows by source: %s",
        _format_counter(Counter(row_source_label(r) for r in sampled_rows)),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name or cfg.tokenizer_name or cfg.model_name,
        trust_remote_code=True,
    )

    train_dataset = FTPODataset(
        sampled_rows,
        tokenizer,
        max_seq_length=cfg.max_seq_length,
        filter_rejected_stop_words=cfg.filter_rejected_stop_words,
        chosen_regularisation_strength=cfg.chosen_regularisation_strength,
    )
    logger.info(
        "tokenized FTPO dataset kept %d/%d sampled rows; filter counts: %s",
        len(train_dataset), len(sampled_rows),
        _format_counter(train_dataset.filter_counts, limit=12),
    )

    effective_batch = cfg.batch_size * cfg.gradient_accumulation_steps
    estimated_steps = math.ceil(len(train_dataset) * cfg.num_epochs / effective_batch)
    logger.info(
        "training schedule: examples=%d, batch=%d, grad_accum=%d, "
        "effective_batch=%d, epochs=%s, estimated_steps=%d",
        len(train_dataset), cfg.batch_size, cfg.gradient_accumulation_steps,
        effective_batch, cfg.num_epochs, estimated_steps,
    )

    cfg.merged_output_dir = resolve_merged_output_dir(cfg, train_examples=len(train_dataset))
    if cfg.merge_lora:
        logger.info("merged model output directory: %s", cfg.merged_output_dir)

    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, quantization_config=quant, device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            device_map={"": 0}, trust_remote_code=True,
        )

    lora_cfg = _make_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)
    if cfg.freeze_early_layers:
        _freeze_lora_to_last_k_layers(model, k=cfg.n_layers_unfrozen, target_modules=tuple(cfg.target_modules))
        log_trainable_lora_by_layer(model)
    model.train()
    model.print_trainable_parameters()

    init_params = inspect.signature(DPOTrainer.__init__).parameters
    processing_kw = {"processing_class" if "processing_class" in init_params else "tokenizer": tokenizer}

    trainer = FTPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_dataset,
        **processing_kw,
        args=_make_dpo_config(cfg),
    )

    trainer.lambda_mse_target = cfg.lambda_mse_target
    trainer.tau_mse_target = cfg.tau_mse_target
    trainer.lambda_mse = cfg.lambda_mse
    trainer.clip_epsilon_logits = cfg.clip_epsilon_logits
    if cfg.early_stopping_chosen_win is not None:
        trainer.add_callback(
            ThresholdStop("chosen_win", threshold=cfg.early_stopping_chosen_win, higher_is_better=True)
        )

    trainer.train()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("saved LoRA adapter to %s", cfg.output_dir)
    _save_merged_model(trainer, tokenizer, cfg)

    return cfg.output_dir, cfg.merged_output_dir if cfg.merge_lora else None
