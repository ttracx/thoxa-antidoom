"""CLI entry point for the THOXA Antidoom pipeline.

Usage:
    thoxa-antidoom -c configs/default.yaml -r runs/antidoom1 \\
        --temp 0.01 \\
        --model-name <hf-model-id>

Supports three modes:
    --generate-only   : run generation + mining, skip training
    --train-only      : train on an existing pairs file, skip generation
    (default)         : run both generation and training

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from thoxa_antidoom.config import load_config


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thoxa-antidoom",
        description="Generate FTPO pairs and train a LoRA adapter to eliminate doom loops.",
    )
    parser.add_argument("-c", "--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("-r", "--run-dir", type=Path, default=Path("runs/antidoom1"), help="output run directory")
    parser.add_argument("--model-name", type=str, default=None, help="override model_name from config or CLI")
    parser.add_argument("--temp", type=float, default=None, help="override generation temperature")
    parser.add_argument("--generate-only", action="store_true", help="run generation + mining only")
    parser.add_argument("--train-only", action="store_true", help="train on existing pairs only")
    parser.add_argument("--pairs-file", type=Path, default=None, help="existing FTPO pairs JSONL for --train-only")
    parser.add_argument("--max-pairs", type=int, default=None, help="override target_pairs")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    cfg = load_config(args.config)

    model_name = args.model_name or cfg.model_name or cfg.generation.model_name
    if not model_name:
        parser.error("model_name must be set via --model-name, config.model_name, or config.generation.model_name")

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.temp is not None:
        cfg.generation.temperature = args.temp
    if args.max_pairs is not None:
        cfg.generation.target_pairs = args.max_pairs

    cfg.train.model_name = model_name
    cfg.generation.model_name = model_name

    do_generate = not args.train_only
    do_train = not args.generate_only

    pairs_path: Path | None = args.pairs_file

    if do_generate:
        from thoxa_antidoom.generate import run_generation
        logging.info("=== Phase 1: Generation + Doom-Loop Mining ===")
        cfg.generation.output_jsonl = run_dir / "iter_0_ftpo_pairs.jsonl"
        cfg.generation.generations_jsonl = run_dir / "iter_0_generations.jsonl"
        _, pairs_path, n_pairs = run_generation(cfg.generation, run_dir, model_name)
        logging.info("collected %d FTPO pairs at %s", n_pairs, pairs_path)

    if do_train:
        from thoxa_antidoom.ftpo_train import run_training
        logging.info("=== Phase 2: FTPO Training ===")
        if pairs_path is None:
            pairs_path = run_dir / "iter_0_ftpo_pairs.jsonl"
        if not pairs_path.exists():
            logging.error("pairs file not found: %s", pairs_path)
            return 1
        cfg.train.dataset_jsonl = pairs_path
        cfg.train.output_dir = run_dir / "ftpo_lora"
        cfg.train.merged_output_dir = run_dir / "ftpo_merged"
        adapter_dir, merged_dir = run_training(cfg.train, pairs_path)
        logging.info("training complete: adapter=%s, merged=%s", adapter_dir, merged_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
