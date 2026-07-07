# THOXA Antidoom

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-0a9ed4.svg)](https://docs.pytest.org)
[![vLLM](https://img.shields.io/badge/inference-vLLM-76b900.svg)](https://docs.vllm.ai)
[![LoRA](https://img.shields.io/badge/adapter-LoRA%20%2B%20PEFT-7c3aed.svg)](https://github.com/huggingface/peft)
[![Built by THOXA](https://img.shields.io/badge/built%20by-THOX.ai%20LLC-00d4aa.svg)](https://thox.ai)

Targeted Final Token Preference Optimization (FTPO) training to eliminate doom
loops in reasoning models. This is a THOXA-adapted implementation of the
Antidoom method published by Liquid AI (July 2026).

> Copyright (c) 2026 Thox.ai LLC. All rights reserved.
> CTO: Tommy Xaypanya | CEO: Craig Ross
> Thox.ai LLC, Texas, USA

---

## What It Does

Doom loops are repetitive degeneration during inference: the model emits a span
(often "Wait, let me reconsider..."), then repeats it until the context window
is exhausted. This is most common in small reasoning models on hard problems at
low temperature.

Antidoom attacks the failure at the exact token where the loop begins:

1. **Generate** completions on a prompt mix at low temperature with top-k
   logprobs.
2. **Detect** inner repetition (a span repeating 4+ times over 60+ chars).
3. **Extract** an FTPO pair at the loop-starting token: the prompt prefix up to
   that position, the rejected token (the loop starter), and up to 20 chosen
   alternative tokens from the model's own top-k at that position.
4. **Train** a LoRA adapter with a single-token preference loss that pushes the
   model to prefer coherent alternatives over the loop-starting token, while an
   MSE tether keeps the rest of the distribution close to the reference.
5. **Merge** the adapter into the base model.

The training teaches the model nothing new about math or code. It removes the
failure mode that was preventing the model from reaching answers it could
already produce.

## Results (Liquid AI self-reported)

| Model | Before | After | Notes |
|---|---|---|---|
| LFM2.5-2.6B (early ckpt) | 10.2% loop rate | 1.4% | Eval scores improved across the board |
| Qwen3.5-4B | 22.9% loop rate | ~1% | Under greedy sampling, evals increased markedly |

Once loops are removed, near-greedy sampling gives the best eval scores,
challenging the "higher temperature helps reasoning" intuition.

## Quick Start

### Prerequisites

- Python 3.11+
- A CUDA or ROCm GPU (for generation + training)
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone https://github.com/ttracx/thoxa-antidoom.git
cd thoxa-antidoom
uv sync
```

### Run

Start a vLLM server for your model:

```bash
vllm serve <your-model-id> --dtype bfloat16 --gpu-memory-utilization 0.85
```

Run the full generate-and-train flow:

```bash
uv run thoxa-antidoom -c configs/default.yaml -r runs/antidoom1 \
  --temp 0.01 \
  --model-name <your-model-id>
```

This writes generated completions and FTPO pairs to `runs/antidoom1/`, trains a
LoRA adapter on the pair file, and writes the merged model under the same run
directory.

### Generate only

```bash
uv run thoxa-antidoom -c configs/default.yaml -r runs/antidoom1 \
  --generate-only --model-name <your-model-id>
```

### Train only (on existing pairs)

```bash
uv run thoxa-antidoom -c configs/default.yaml -r runs/antidoom1 \
  --train-only --pairs-file runs/antidoom1/iter_0_ftpo_pairs.jsonl \
  --model-name <your-model-id>
```

### Run tests

```bash
uv run pytest tests/ -v
```

## How It Works

### Doom loop detection

The detector (`repetition.py`) samples short 16-character fingerprints at
128-character intervals across the completion. When the same fingerprint
appears at two positions, the distance between them is a candidate period. The
detector then verifies that the pattern at that period repeats at least
`min_repeats` times and covers at least `min_total_repeated` characters.

A loop is flagged when a section repeats at least 4 times over at least 60
characters. The most common loop-starting tokens on LFM2.5-2.6B were:

| Token | Share of loops |
|---|---|
| ` the` | 11.39% |
| ` So` | 4.51% |
| `Alternatively` | 3.22% |
| `Wait` | 2.56% |
| ` But` | 2.46% |

These overtrained discourse markers become attractive fallback continuations
when the model is uncertain or stuck, restarting the same local reasoning
pattern instead of making progress.

### FTPO pair extraction

Once a loop is found, the detector returns the character offset where the
second occurrence begins (the repeat start). This is mapped to a token index via
the `TokenState` running text buffer. The token at that index is the rejected
token. The model's top-k logprobs at that position are filtered (removing the
rejected token, short tokens, non-alphanumeric noise) to produce up to 20
chosen alternatives.

Each FTPO row is:

```json
{
  "context_before": "<prompt + all tokens before the rejected token>",
  "rejected_token": " Wait",
  "rejected_decoded": " Wait",
  "multi_chosen_tokens": [" Therefore", " The"],
  "multi_chosen_decoded": [" Therefore", " The"],
  "repetition": {
    "start_char": 1024,
    "repeat_start_char": 1526,
    "rejected_token_index": 342
  }
}
```

### FTPO loss

The trainer (`reference_ftpo_trainer.py`) subclasses TRL's DPOTrainer but
replaces the loss with a **single-token margin-based preference loss**:

- The model is run on the prompt prefix (right-padded so the last position is
  the prediction point).
- At the final position, the logits for the rejected token and each chosen
  alternative are compared in **raw logit space** (no softmax).
- `delta = logit_chosen - logit_rejected` for each chosen token.
- Preference loss: `softplus((epsilon - delta) / tau)` weighted by
  `clamp((epsilon - delta) / epsilon, 0, 1)`.
- An MSE tether on the rest of the vocabulary keeps the distribution close to
  the reference model (`lambda_mse`).
- A targeted MSE on chosen/rejected tokens allows them to move more freely (up
  to `tau_mse_target`) before the penalty kicks in (`lambda_mse_target`).

Early stopping fires when `chosen_win` (fraction of chosen tokens beating the
rejected token) crosses the configured threshold, preventing over-training.

> **Note:** The blog describes stopping at `chosen_win=0.35`. The shipped
> `configs/default.yaml` uses `0.4` (slightly more conservative). Tune based on
> your model's behavior.

### Data regularisation

Before training, the FTPO pairs are regularised to prevent over-represented
loop-starting tokens (like `Wait` or `Alternatively`) from dominating the
training set. Rejected-token normalisation flattens above-median token counts
toward the median. Optional source balancing (`sqrt` or `equal` mode)
downsamples over-represented prompt sources.

## Configuration

All knobs live in `configs/default.yaml`. The most common fields to change:

| Parameter | Default | Notes |
|---|---|---|
| `train.lora_r` | 256 | Higher rank = better learnability, less degradation |
| `train.learning_rate` | 1.5e-5 | Range 4e-6 to 2e-5; over-training is easy |
| `train.early_stopping_chosen_win` | 0.4 | Stop when chosen tokens win this fraction |
| `train.lambda_mse` | 0.4 | Tether strength on non-chosen/rejected vocab |
| `train.lambda_mse_target` | 0.05 | Tether on chosen/rejected tokens |
| `train.tau_mse_target` | 0.5 | Allowed logit movement before tether penalty |
| `train.clip_epsilon_logits` | 2.0 | Margin target for chosen vs rejected |
| `generation.temperature` | 0.1 | Low temp elicits loops; near-greedy is best |
| `generation.repetition.min_repeats` | 4 | Minimum repetitions to flag a loop |
| `generation.repetition.min_total_repeated` | 60 | Min chars covered by repeats |
| `generation.target_pairs` | 20000 | Pairs to collect before stopping generation |

## Iterative Application

After one round of Antidoom, the doom-loop rate drops but new failure points
can surface where other tokens now trigger loops. Applying additional rounds
targets these newly surfaced loops, further reducing the rate. Run the pipeline
again with the merged model from the previous round as the new base:

```bash
# Round 2: use round 1's merged model as the base
uv run thoxa-antidoom -c configs/default.yaml -r runs/antidoom2 \
  --temp 0.01 \
  --model-name runs/antidoom1/ftpo_merged/<merged-model-name>
```

## Post-Quantization Validation (Critical)

FTPO makes small, surgical logit adjustments at specific token positions.
Aggressive quantization (Q4_K_M) can introduce noise comparable to or larger
than those shifts. **Always validate the doom-loop rate after quantization on
the target device**, not just on the full-precision merged model.

If quantization undoes the fix:
1. Try Q5_K_M or Q6_K quantization tiers
2. Re-measure loop rate on-target
3. Adjust your on-device memory budget accordingly

## Project Structure

```
thoxa-antidoom/
  AGENTS.md                # agent instructions + technical notes
  LICENSE                  # Apache 2.0 (Thox.ai LLC)
  README.md                # this file
  pyproject.toml           # package config + dependencies
  configs/
    default.yaml           # generation + training config
  src/
    thoxa_antidoom/
      __init__.py
      repetition.py        # doom loop detector
      tokens.py            # token decoding + char-to-token mapping
      sampling.py          # chosen-token selection at rejection point
      prompts.py           # prompt dataset loading (HF / JSONL)
      client.py            # vLLM / OpenAI completions client
      generate.py          # generation + mining pipeline
      ftpo_data.py         # FTPO dataset, row sampling, regularisation
      reference_ftpo_trainer.py  # FTPO loss (single-token preference + MSE)
      ftpo_train.py        # training orchestration (LoRA + merge)
      config.py            # YAML config dataclasses
      cli.py               # CLI entry point
  tests/
    test_repetition.py     # detector unit tests
  docs/
    AGENT_HANDOFF.md       # comprehensive agent handoff
    INTEGRATION_GUIDE.md   # integration guide for THOXA stack
```

## Upstream Attribution

This project adapts:
- **Antidoom** by Liquid AI (blog: https://www.liquid.ai/blog/antidoom,
  code: https://github.com/Liquid4All/antidoom, Apache 2.0)
- **Antislop / FTPO** by Paech, Roush, Goldfeder, Shwartz-Ziv
  (arXiv:2510.15061, code: https://github.com/sam-paech/auto-antislop, MIT)

## Citation

```yaml
@article{liquidAI2026Antidoom,
    author = {Liquid AI},
    title = {Reducing Doom Loops with Final Token Preference Optimization},
    journal = {Liquid AI Blog},
    year = {2026},
    note = {www.liquid.ai/blog/antidoom}
}
```

## License

Apache 2.0. Copyright (c) 2026 Thox.ai LLC.
