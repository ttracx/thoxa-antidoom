# Integration Guide: THOXA Antidoom in the Edge AI Stack

> Copyright (c) 2026 Thox.ai LLC. All rights reserved.
> CTO: Tommy Xaypanya | CEO: Craig Ross

## Where Antidoom lands

Antidoom is an **offline model-preparation step**. It runs in CI / model-build
pipelines, not on the device. The device only ever receives the corrected
weights (as a GGUF file). No runtime, kernel, or mesh changes are required.

```
CI / Model-Build Pipeline
  |
  +-- Base model checkpoint
  |     |
  |     v
  +-- [Antidoom] generate + mine + train + merge
  |     |
  |     v
  +-- Corrected full-precision model
  |     |
  |     v
  +-- Quantize (GGUF Q4_K_M / Q5_K_M)
  |     |
  |     v
  +-- Post-quantization loop-rate validation (HARD GATE)
  |     |
  |     v
  +-- Ship corrected GGUF to device
```

## Stack layer mapping

| Layer | Antidoom role | Integration needed |
|---|---|---|
| On-device inference (llama.cpp/GGUF) | Primary beneficiary | None - just ship corrected weights |
| Agent orchestration / swarm | Complementary | Cleaner base model reduces wasted agent turns |
| OS/kernel (Rust microkernel) | No role | Loads GGUF as before |
| Distributed encrypted mesh (WireGuard) | No direct role | Bounded generation improves scheduling estimates |

## Concrete integration steps

### Step 1: Measure baseline loop rate

Before applying Antidoom, measure your model's doom-loop rate on a
representative eval prompt mix. Use the detector directly:

```python
from thoxa_antidoom.repetition import find_inner_repetition

loop_count = 0
total = 0
for completion in your_eval_completions:
    found, _ = find_inner_repetition(completion)
    if found:
        loop_count += 1
    total += 1

loop_rate = loop_count / total
```

**Threshold to proceed:** loop rate above 2-3%. Below that, an inference-time
backstop (DRY sampling, repetition penalty) may suffice.

### Step 2: Prepare domain-specific prompts

The default `antidoom-mix-v1.0` dataset is math/code-heavy. For regulated
verticals (healthcare, legal, finance, defense), build a domain-specific
loop-elicitation prompt set. This avoids distribution shift and keeps
compliance reviewers comfortable with training data provenance.

Create a JSONL file with prompts in ShareGPT or plain-text format:

```json
{"conversations": [{"from": "human", "value": "Your domain prompt here"}], "source": "healthcare"}
```

Point the config at it:

```yaml
generation:
  input_jsonl: "data/domain_prompts.jsonl"
  prompt_field: "conversations"
  hf_dataset: null
```

### Step 3: Run the pipeline

```bash
vllm serve <your-model-id> --dtype bfloat16 --gpu-memory-utilization 0.85

uv run thoxa-antidoom -c configs/default.yaml -r runs/antidoom1 \
  --temp 0.01 \
  --model-name <your-model-id>
```

### Step 4: Merge and quantize

```bash
# The pipeline auto-merges. Find the merged model:
ls runs/antidoom1/ftpo_merged/

# Convert to GGUF:
python convert_hf_to_gguf.py runs/antidoom1/ftpo_merged/<model-name> --outtype f16

# Quantize:
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

### Step 5: Validate on-target (HARD GATE)

Run your eval suite on the quantized model **on the actual edge device**. Check:

1. Doom-loop rate (should be < 2%)
2. Accuracy on prompts that previously looped
3. Standard eval scores (should match or improve vs. baseline)
4. Latency and memory footprint

If loop rate regressed after quantization, try Q5_K_M or Q6_K.

### Step 6: Ship and iterate

Ship the validated GGUF. If new loops surface in production, run a second
Antidoom round with the merged model from round 1 as the new base.

## CI pipeline integration

Add Antidoom as a post-training step in your model build CI:

```yaml
# .github/workflows/model-build.yml (sketch)
- name: Antidoom FTPO pass
  run: |
    vllm serve $MODEL_ID --dtype bfloat16 &
    sleep 60
    uv run thoxa-antidoom -c configs/default.yaml -r runs/antidoom \
      --temp 0.01 --model-name $MODEL_ID
- name: Quantize
  run: |
    python convert_hf_to_gguf.py runs/antidoom/ftpo_merged/*/ --outtype f16
    ./llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
- name: Validate loop rate on-target
  run: python scripts/validate_loop_rate.py model-q5_k_m.gguf
```

## Inference-time backstop (belt and suspenders)

During initial rollout, keep DRY sampling / repetition penalty enabled as a
runtime backstop. Once FTPO-corrected weights prove stable in production,
dial it down:

```bash
# llama.cpp with DRY backstop
./llama-cli -m model-q5_k_m.gguf \
  --dry-multiplier 0.5 --dry-base 1.0 \
  --temp 0.1 --top-k 50 --min-p 0.01
```

## License posture

- **Antidoom code** (this repo + Liquid4All/antidoom): Apache 2.0
- **Antislop/FTPO trainer** (sam-paech/auto-antislop): MIT
- **antidoom-mix-v1.0 dataset**: Apache 2.0
- **LFM2/LFM2.5 model weights**: Liquid LFM Open License (separate, has a
  $10M annual revenue commercial-use threshold)

All three tooling assets are permissive with no copyleft and no revenue cap.
The LFM weight license only applies if you use Liquid's model weights. Since
the technique is portable (proven on Qwen3.5-4B), you can capture the doom-loop
fix with your own base models and never touch the LFM weight license.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Over-training creates new loops | Early stopping on `chosen_win` (mandatory) |
| Quantization washes out FTPO gains | Post-quantization validation gate (Q5/Q6 fallback) |
| Distribution shift from math/code prompts | Domain-specific loop-elicitation prompt set |
| LFM2 non-standard LoRA targets | Consult Liquid fine-tuning docs for LFM weights |
| Removing loops masks reasoning failures | Check accuracy on previously-looping prompts |
