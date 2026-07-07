# THOXA Antidoom - Agent Instructions

## Project identity

- **Name:** thoxa-antidoom
- **Owner:** Thox.ai LLC (Texas, USA)
- **CTO:** Tommy Xaypanya
- **CEO:** Craig Ross
- **License:** Apache 2.0
- **Purpose:** Targeted FTPO training to eliminate doom loops in reasoning models

## What this repo does

Implements a THOXA-adapted version of Liquid AI's Antidoom method for
eliminating repetitive degeneration ("doom loops") in small reasoning
models via Final Token Preference Optimization (FTPO).

The pipeline has two phases:
1. **Generate + mine:** Generate completions at low temperature, detect
   inner repetition, extract single-token preference pairs at the
   loop-starting position.
2. **Train + merge:** Train a LoRA adapter with the FTPO loss
   (softplus margin + MSE tether), then merge into the base model.

## Architecture

```
src/thoxa_antidoom/
  repetition.py             # doom loop detector (fingerprint sampling)
  tokens.py                 # token decoding + char-to-token mapping
  sampling.py               # chosen-token selection at rejection point
  prompts.py                # prompt dataset loading (HF / JSONL)
  client.py                 # vLLM / OpenAI completions client
  generate.py               # generation + mining pipeline
  ftpo_data.py              # FTPO dataset, row sampling, regularisation
  reference_ftpo_trainer.py # FTPO loss (single-token preference + MSE)
  ftpo_train.py             # training orchestration (LoRA + merge)
  config.py                 # YAML config dataclasses
  cli.py                    # CLI entry point
```

## Key technical notes

- The FTPO loss is NOT a KL divergence. It is a softplus-margin loss on
  raw logits: `softplus((epsilon - delta) / tau)` where
  `delta = logit_chosen - logit_rejected`, plus a two-part MSE tether.
- Early stopping on `chosen_win` is mandatory. Over-training creates new
  loops. The blog says stop at 0.35; the shipped default config uses 0.4.
- LoRA rank 128-256 is recommended. Higher rank = better learnability
  with less degradation.
- Post-quantization validation is a hard gate. FTPO makes small targeted
  logit shifts that aggressive quantization (Q4_K_M) can wash out.
- For LFM2/LFM2.5 hybrid architectures, LoRA target modules differ from
  standard transformer projections (LIV convolution layers). Consult
  Liquid's fine-tuning docs before fine-tuning LFM weights.

## Development rules

1. Never omit implementation details for brevity.
2. Full implementations only - no placeholder comments.
3. Include robust error handling.
4. Follow the existing module structure.
5. Keep copyright headers on all source files.
6. Tests go in `tests/` and run with `PYTHONPATH=src pytest tests/`.

## Running

```bash
uv sync
vllm serve <model-id> --dtype bfloat16 --gpu-memory-utilization 0.85
uv run thoxa-antidoom -c configs/default.yaml -r runs/antidoom1 \
  --temp 0.01 --model-name <model-id>
```

## Safety

- Never push to remote without operator confirmation.
- Never open a PR without operator review.
- Hardware-touching tasks require operator confirmation.
- Working branches use prefix `<agent-name>/<task-slug>`.
