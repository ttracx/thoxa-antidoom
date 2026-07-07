# Ecosystem Map: THOXA Antidoom

> Copyright (c) 2026 Thox.ai LLC. All rights reserved.
> CTO: Tommy Xaypanya | CEO: Craig Ross

## Architecture

```
                     +-----------------------+
                     |   Prompt Data Source   |
                     |  (HF dataset / JSONL)  |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------+-----------+
                     |   Generation Client    |
                     |  (vLLM HTTP / OpenAI)  |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------+-----------+
                     |  Doom Loop Detector    |
                     |  (repetition.py)       |
                     +-----------+-----------+
                                 |
                          found? | yes
                                 v
                     +-----------+-----------+
                     |  FTPO Pair Extractor   |
                     |  (tokens + sampling)   |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------+-----------+
                     |  Row Regularisation    |
                     |  (ftpo_data.py)        |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------+-----------+
                     |  FTPO LoRA Trainer     |
                     |  (reference trainer)   |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------+-----------+
                     |  Adapter Merge         |
                     |  (ftpo_train.py)       |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------+-----------+
                     |  Quantize + Validate   |
                     |  (external: GGUF)      |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------------------+
                     |  Corrected Edge Model  |
                     |  (GGUF on device)      |
                     +-----------------------+
```

## Services and modules

| Module | Type | Purpose |
|---|---|---|
| `repetition.py` | Library | Detect doom loops in completion text |
| `tokens.py` | Library | Token decoding and char-to-token mapping |
| `sampling.py` | Library | Select chosen alternative tokens |
| `prompts.py` | Library | Load prompts from HF or local JSONL |
| `client.py` | Library | HTTP client for vLLM/OpenAI completions |
| `generate.py` | Pipeline | Orchestrate generation + mining |
| `ftpo_data.py` | Library | Dataset, sampling, regularisation |
| `reference_ftpo_trainer.py` | Library | FTPO loss + collator |
| `ftpo_train.py` | Pipeline | Training orchestration, LoRA, merge |
| `config.py` | Library | YAML config dataclasses |
| `cli.py` | CLI | Entry point for full pipeline |

## External dependencies

| Dependency | Purpose | License |
|---|---|---|
| PyTorch | Tensor operations, autograd | BSD |
| Transformers | Model loading, tokenization | Apache 2.0 |
| TRL | DPOTrainer base class | Apache 2.0 |
| PEFT | LoRA adapter management | Apache 2.0 |
| vLLM | Efficient generation with logprobs | Apache 2.0 |
| datasets | HuggingFace dataset loading | Apache 2.0 |
| httpx | HTTP client for completions API | BSD |
| PyYAML | Config file parsing | MIT |
| numpy | Numerical operations in regularisation | BSD |

## External integrations

| Integration | Direction | Protocol |
|---|---|---|
| vLLM server | Outbound | HTTP /v1/completions |
| HuggingFace Hub | Outbound | datasets API |
| llama.cpp / GGUF | Inbound (downstream) | File-based (GGUF) |
| Liquid4All/antidoom | Upstream reference | Apache 2.0 |
| sam-paech/auto-antislop | Upstream reference | MIT |
| LiquidAI/antidoom-mix-v1.0 | Data source | Apache 2.0 |
