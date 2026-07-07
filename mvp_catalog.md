# MVP Catalog: THOXA Antidoom

> Copyright (c) 2026 Thox.ai LLC. All rights reserved.
> CTO: Tommy Xaypanya | CEO: Craig Ross

## MVP-1: Doom Loop Detector

- **Purpose:** Detect repetitive degeneration in model completions
- **Scope:** Text-only fingerprint-based repetition detection
- **Dependencies:** None (pure Python)
- **Status:** Complete, tested (9 unit tests passing)

## MVP-2: FTPO Pair Builder

- **Purpose:** Extract single-token preference pairs from detected loops
- **Scope:** Token mapping, chosen-token selection, pair formatting
- **Dependencies:** MVP-1, vLLM generation client, tokenizer
- **Status:** Complete

## MVP-3: Generation + Mining Pipeline

- **Purpose:** Batch-generate completions, detect loops, collect FTPO pairs
- **Scope:** Prompt loading, threaded generation, pair extraction, JSONL output
- **Dependencies:** MVP-2, vLLM server, HuggingFace datasets
- **Status:** Complete

## MVP-4: FTPO Trainer

- **Purpose:** Train LoRA adapter with single-token preference loss
- **Scope:** Dataset tokenization, row regularisation, FTPO loss, early stopping
- **Dependencies:** PyTorch, Transformers, TRL, PEFT, MVP-2
- **Status:** Complete

## MVP-5: Merge + Ship

- **Purpose:** Merge LoRA adapter into base model, prepare for quantization
- **Scope:** Adapter merge, model save, descriptive naming
- **Dependencies:** MVP-4, PEFT
- **Status:** Complete

## MVP-6: Iterative Rounds

- **Purpose:** Run multiple Antidoom rounds to catch newly surfaced loops
- **Scope:** Use merged model from round N as base for round N+1
- **Dependencies:** MVP-5
- **Status:** Complete (manual orchestration via CLI)

## MVP-7: Post-Quantization Validation (Planned)

- **Purpose:** Validate doom-loop rate after GGUF quantization on target device
- **Scope:** Load quantized model, run eval, compare loop rate
- **Dependencies:** llama.cpp, GGUF tooling
- **Status:** Not started - documented in INTEGRATION_GUIDE.md

## MVP-8: Domain-Specific Prompt Sets (Planned)

- **Purpose:** Build loop-elicitation prompts for healthcare/legal/finance/defense
- **Scope:** Curated prompt JSONL files per vertical
- **Dependencies:** Domain expertise, compliance review
- **Status:** Not started
