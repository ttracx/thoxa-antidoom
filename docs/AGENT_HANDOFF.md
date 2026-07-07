# Agent Handoff: THOXA Antidoom

> Copyright (c) 2026 Thox.ai LLC. All rights reserved.
> CTO: Tommy Xaypanya | CEO: Craig Ross

## Document purpose

This handoff gives any incoming agent or engineer the full context needed to
understand, extend, operate, and debug the THOXA Antidoom pipeline without
reading the entire codebase first.

## 1. System overview

THOXA Antidoom is a two-phase offline training pipeline that eliminates doom
loops (repetitive degeneration) in reasoning models:

```
Prompt mix --> [Generate at low temp] --> [Detect loops] --> [Extract FTPO pairs]
                                                                        |
                                                                        v
            [Merge LoRA] <-- [Train LoRA adapter] <-- [Regularise pairs]
                 |
                 v
            Validated corrected model (GGUF)
```

### Phase 1: Generation + Mining (`generate.py`)

- Loads prompts from HuggingFace (`LiquidAI/antidoom-mix-v1.0`) or local JSONL
- Sends each prompt to a vLLM server at low temperature with top-k logprobs
- Scans each completion for inner repetition via `repetition.find_inner_repetition`
- On loop detection, maps the repeat-start char offset to a token index
- Extracts the rejected token (loop starter) and up to 20 chosen alternatives
- Writes `iter_0_generations.jsonl` and `iter_0_ftpo_pairs.jsonl`
- Stops after collecting `target_pairs` (default 20,000)

### Phase 2: Training + Merge (`ftpo_train.py`)

- Loads FTPO pairs from JSONL
- Filters trimmed prompts, enforces `min_chosen_tokens`
- Regularises rejected-token distribution (flattens over-represented starters)
- Tokenizes into `(prompt_ids, chosen_ids, rejected_token_id)` tuples
- Trains a LoRA adapter with the FTPO loss for 1 epoch
- Early stops on `chosen_win` threshold
- Merges adapter into base model, saves to disk

## 2. Module responsibilities

| Module | Responsibility | Depends on torch? |
|---|---|---|
| `repetition.py` | Detect doom loops in text | No |
| `tokens.py` | Token decoding, char-to-token mapping | No |
| `sampling.py` | Select chosen alternative tokens | No |
| `trimmed.py` | Mark/detect trimmed source prompts | No |
| `prompts.py` | Load prompts from HF or JSONL | No (uses `datasets` lazily) |
| `client.py` | HTTP client for vLLM/OpenAI completions | No (uses `httpx`) |
| `config.py` | YAML config dataclasses | No (uses `yaml`) |
| `generate.py` | Orchestrate generation + mining | No (orchestration only) |
| `ftpo_data.py` | Dataset, sampling, regularisation | Yes |
| `reference_ftpo_trainer.py` | FTPO loss + collator | Yes |
| `ftpo_train.py` | Training orchestration, LoRA, merge | Yes |
| `cli.py` | CLI entry point | No (imports lazily) |

## 3. The FTPO loss in detail

The loss operates only on the **final position** of the right-padded prompt
prefix. The model produces `logits_last` (shape `[B, vocab]`).

### Preference loss

```python
delta_tok = logits_last[:, chosen_ids] - logits_last[:, rejected_id]
weights = clamp((epsilon - delta_tok) / epsilon, 0, 1) * chosen_mask
per_tok_loss = softplus((epsilon - delta_tok) / tau)
pref_loss = mean over batch of sum_chosen(per_tok_loss * weights) / n_chosen
```

- `epsilon` (`clip_epsilon_logits`, default 2.0): target logit margin
- `tau`: temperature for softplus, fixed at 1.0
- `weights`: down-weights tokens that already have a large margin (no need
  to push them further)

### MSE tether (two-part)

1. **Vocab tether** (`lambda_mse=0.4`): MSE between current and reference
   logits on all vocab tokens *except* chosen and rejected. Keeps the rest
   of the distribution pinned.

2. **Target tether** (`lambda_mse_target=0.05`, `tau_mse_target=0.5`): MSE
   on chosen and rejected tokens only, but with an allowance: only
   penalizes logit movement *exceeding* `tau_mse_target`. This lets the
   chosen/rejected logits move freely up to the threshold.

### Reference model

When using LoRA (which is the default), `ref_model=None` and the reference
logits are computed by disabling the adapter via
`_ftpo_null_ref_context`. No separate reference model is loaded.

### Metrics logged

- `pref_loss`: the preference component
- `chosen_win`: fraction of chosen tokens with higher logprob than rejected
- `margin_win`: fraction of chosen tokens with delta >= epsilon
- `mean_delta`, `median_delta`: logit margin statistics
- `mse_elem`, `mse_tgt_tokenwise`: tether components

## 4. Critical operational notes

### Over-training is the #1 failure mode

The blog and the code both warn: training longer than necessary degrades the
model and can create *new* doom loops. The early-stopping callback on
`chosen_win` is not optional. If you disable it, you will over-train.

The blog says stop at 0.35. The shipped config uses 0.4. For a new model,
start at 0.35 and watch `chosen_win` in the logs. If it plateaus before
0.35, stop there.

### Post-quantization validation is mandatory

FTPO makes small logit adjustments. Quantization to Q4_K_M can introduce
noise larger than those adjustments. The validation sequence must be:

1. Measure loop rate on full-precision merged model
2. Quantize to target tier (Q4_K_M, Q5_K_M, etc.)
3. Measure loop rate on quantized model **on the target device**
4. If loop rate regressed, try a higher precision tier

If you skip step 3, you may ship a model that loops in production despite
passing full-precision eval.

### Compute scales with loop rate

Generation time is inversely proportional to the model's doom-loop rate. The
pipeline stops after collecting `target_pairs` pairs, not after exhausting
prompts. A 10% loop rate model needs ~200k prompts for 20k pairs. A 2% loop
rate model needs ~1M prompts. Budget accordingly.

### LFM2 hybrid architecture caveat

If fine-tuning Liquid LFM2/LFM2.5 weights, the LoRA target modules in
`default.yaml` (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj, lm_head`) are standard transformer modules. LFM2 uses Linear
Input-Varying (LIV) convolution layers that do not have these projections.
Consult Liquid's fine-tuning documentation for the correct target module
names before training on LFM weights.

## 5. Iterative rounds

After one round of Antidoom, the loop rate drops but new failure points can
surface. Run a second round with the merged model from round 1 as the new
base. In practice, 2-3 rounds may be needed to get below 1%.

```
Round 1: base_model --> antidoom1 --> merged1 (loop rate: ~5%)
Round 2: merged1    --> antidoom2 --> merged2 (loop rate: ~2%)
Round 3: merged2    --> antidoom3 --> merged3 (loop rate: ~1%)
```

Each round generates a fresh set of pairs because the failure points shift.

## 6. Data flow and file artifacts

```
runs/antidoomN/
  iter_0_generations.jsonl   # full completions + loop detection metadata
  iter_0_ftpo_pairs.jsonl    # preference pairs (input to training)
  ftpo_lora/                 # trained LoRA adapter
  ftpo_merged/               # merged full model
    <name>_antidoom<size>_from<id>_<N>-samples_lr<X>_ep<Y>_<job>_HF/
```

### FTPO pair JSONL schema

```json
{
  "context_before": "string",
  "rejected_token": "string",
  "rejected_decoded": "string",
  "multi_chosen_tokens": ["string", ...],
  "multi_chosen_decoded": ["string", ...],
  "source": "string",
  "source_prompt": "string",
  "repetition": {
    "start_char": int,
    "repeat_start_char": int,
    "rejected_token_index": int,
    "period": int,
    "repeats": int,
    "snippet": "string"
  }
}
```

## 7. Testing

```bash
PYTHONPATH=src pytest tests/ -v
```

Currently tests the repetition detector only (9 tests). The torch-dependent
modules require a GPU environment to test meaningfully. Recommended additions:
- `test_sampling.py`: chosen-token selection edge cases
- `test_ftpo_data.py`: row sampling and regularisation
- `test_config.py`: YAML loading and validation

## 8. Known limitations

1. The repetition detector uses fingerprint sampling with `sample_len=16` and
   `sample_interval=128`. Loops with periods shorter than `sample_len` may be
   missed. Reduce `sample_len` if targeting short-period loops.
2. The generation client uses HTTP (`httpx`) against a vLLM server. The
   upstream repo also supports embedded vLLM (Python API, no server). This
   implementation uses the HTTP path for simplicity.
3. Antidoom fixes token-level repetition only. It does not fix semantic
   loops (stuck on an idea without textual repetition), hallucination, or
   reasoning-quality limitations.
4. Removing a loop does not give the model the ability to solve the problem
   it was stuck on. It will fail differently (possibly with a confident
   wrong answer). For regulated verticals, check accuracy on previously
   looping prompts, not just loop rate.

## 9. Next steps for an incoming agent

1. **Read** `reference_ftpo_trainer.py` to understand the loss math
2. **Run** `PYTHONPATH=src pytest tests/ -v` to confirm the detector works
3. **Review** `configs/default.yaml` and adjust for your target model
4. **Provision** a GPU with vLLM for the generation phase
5. **Execute** the full pipeline on a small test model to validate end-to-end
6. **Validate** post-quantization loop rate on the target device
