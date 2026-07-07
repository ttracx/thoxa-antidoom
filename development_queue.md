# Development Queue: THOXA Antidoom

> Copyright (c) 2026 Thox.ai LLC. All rights reserved.
> CTO: Tommy Xaypanya | CEO: Craig Ross

## Priority scoring

```
Priority Score =
  (Market Value x 0.4) +
  (Technical Feasibility x 0.3) +
  (Time-to-Market x 0.2) +
  (Strategic Importance x 0.1)
```

Scale: 1-10 per factor.

## Backlog

| # | Task | Market | Feasibility | TTM | Strategic | Score | Status |
|---|---|---|---|---|---|---|---|
| 1 | Post-quantization loop-rate validation script | 8 | 9 | 8 | 7 | 8.1 | Not started |
| 2 | Embedded vLLM generation (no HTTP server needed) | 6 | 8 | 7 | 5 | 6.5 | Not started |
| 3 | Multi-round auto-orchestration | 7 | 7 | 6 | 6 | 6.6 | Not started |
| 4 | Domain-specific prompt sets (healthcare, legal, finance, defense) | 8 | 5 | 4 | 8 | 6.5 | Not started |
| 5 | AMD/ROCm config and testing | 5 | 6 | 5 | 5 | 5.3 | Not started |
| 6 | LFM2 hybrid architecture LoRA target module support | 6 | 4 | 3 | 7 | 5.1 | Not started |
| 7 | Semantic loop detection (idea-level repetition without textual match) | 7 | 3 | 2 | 7 | 5.0 | Not started |
| 8 | CI/CD workflow template for model build pipeline | 6 | 8 | 7 | 6 | 6.7 | Not started |
| 9 | Test coverage for sampling.py and ftpo_data.py | 4 | 9 | 8 | 4 | 6.0 | Not started |
| 10 | wandb/tensorboard logging integration | 3 | 9 | 9 | 3 | 5.4 | Not started |

## Next actions

1. Build post-quantization validation script (task #1) - highest priority
2. Add embedded vLLM generation path (task #2) - simplifies CI integration
3. Create CI/CD workflow template (task #8) - enables automated model builds
