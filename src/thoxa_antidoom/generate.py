"""Generation + doom-loop mining pipeline.

For each prompt: generate a completion at low temperature with top-k logprobs,
scan for inner repetition, and if a loop is found, extract an FTPO pair at the
loop-starting token. The pair consists of the prompt prefix up to that token,
the rejected token (loop starter), and up to N chosen alternative tokens from
the model's own top-k at that position.

The pipeline runs in batches, writing generations and FTPO pairs to JSONL files,
and stops once ``target_pairs`` pairs have been collected.

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

import json
import logging
import concurrent.futures
from pathlib import Path
from typing import Any

from thoxa_antidoom.client import GenerationClient
from thoxa_antidoom.config import GenerationConfig
from thoxa_antidoom.prompts import load_prompts
from thoxa_antidoom.repetition import find_inner_repetition
from thoxa_antidoom.sampling import select_chosen_tokens
from thoxa_antidoom.tokens import TokenState, decode_token

logger = logging.getLogger(__name__)


def _format_prompt(text: str, chat_messages: list[dict] | None, template: str) -> str:
    """Apply the prompt template to a plain-text or chat prompt."""
    if chat_messages:
        # For chat models, join messages into a single prompt string.
        parts = []
        for msg in chat_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return template.format(prompt="\n".join(parts))
    return template.format(prompt=text)


def _build_ftpo_pair(
    state: TokenState,
    hit,
    cfg: GenerationConfig,
    source: str,
    raw_prompt: str,
) -> dict[str, Any] | None:
    """Extract a single FTPO preference pair from a detected loop.

    Maps the loop's repeat-start character offset to a token index, identifies
    the rejected token, and selects chosen alternatives from the model's
    top-k logprobs at that position.
    """
    repeat_char = hit.repeat_start
    token_idx = state.char_to_token_index(repeat_char)
    if token_idx is None or token_idx < 0:
        return None

    rejected_raw = state.token_strings[token_idx]
    rejected_decoded = decode_token(rejected_raw)

    # Get the logprob alternatives at the rejection position.
    logprob_alts = state.logprobs.get(token_idx, [])
    if not logprob_alts:
        return None

    chosen_raws, chosen_decoded = select_chosen_tokens(
        logprob_alts,
        rejected_token=rejected_raw,
        temperature=cfg.temperature,
        min_p=cfg.chosen_min_p,
        top_k=cfg.chosen_top_k,
        max_tokens=cfg.chosen_max_tokens,
        min_decoded_chars=cfg.chosen_min_decoded_chars,
        require_alnum=cfg.chosen_require_alnum,
        skip_raw_token_substrings=cfg.chosen_skip_raw_token_substrings,
    )

    if not chosen_raws:
        return None

    context = state.context_before(token_idx)

    return {
        "context_before": context,
        "rejected_token": rejected_raw,
        "rejected_decoded": rejected_decoded,
        "multi_chosen_tokens": chosen_raws,
        "multi_chosen_decoded": chosen_decoded,
        "source": source,
        "source_prompt": raw_prompt,
        "repetition": {
            "start_char": hit.start,
            "repeat_start_char": hit.repeat_start,
            "rejected_token_index": token_idx,
            "period": hit.period,
            "repeats": hit.repeats,
            "snippet": hit.snippet,
        },
    }


def _process_single_prompt(
    client: GenerationClient,
    cfg: GenerationConfig,
    model_name: str,
    prompt_text: str,
    source: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Generate + mine a single prompt. Returns (generation_record, ftpo_pair)."""
    try:
        state = client.generate_state(
            model=model_name,
            prompt=prompt_text,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            top_logprobs=cfg.top_logprobs,
            stop=cfg.stop or None,
        )
    except Exception as exc:
        logger.warning("generation failed for source=%s: %s", source, exc)
        return None, None

    rep_kwargs = cfg.repetition or {}
    found, hit = find_inner_repetition(state.text, **rep_kwargs)

    gen_record = {
        "source": source,
        "prompt": prompt_text,
        "completion": state.text,
        "status": "loop_detected" if found else "clean",
        "repetition": {
            "start_char": hit.start,
            "repeat_start_char": hit.repeat_start,
            "period": hit.period,
            "repeats": hit.repeats,
            "snippet": hit.snippet,
        } if found else None,
    }

    if not found:
        return gen_record, None

    pair = _build_ftpo_pair(state, hit, cfg, source, prompt_text)
    return gen_record, pair


def run_generation(
    cfg: GenerationConfig,
    run_dir: Path,
    model_name: str,
) -> tuple[Path, Path, int]:
    """Run the full generate-and-mine loop.

    Returns ``(generations_path, pairs_path, n_pairs)``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_path = run_dir / "iter_0_generations.jsonl"
    pairs_path = run_dir / "iter_0_ftpo_pairs.jsonl"

    client = GenerationClient(
        base_url=cfg.api_base_url,
        api_key=cfg.api_key,
        timeout=cfg.timeout,
    )

    rep_defaults = {
        "min_repeats": 4,
        "max_period": 1024,
        "min_period": 1,
        "min_total_repeated": 60,
        "sample_len": 16,
        "sample_interval": 128,
    }
    rep_defaults.update(cfg.repetition or {})
    cfg.repetition = rep_defaults

    temperatures = cfg.temperatures or [cfg.temperature]

    n_pairs = 0
    n_prompts = 0
    n_loops = 0

    gen_fh = gen_path.open("w", encoding="utf-8")
    pairs_fh = pairs_path.open("w", encoding="utf-8")

    try:
        prompt_iter = load_prompts(
            hf_dataset=cfg.hf_dataset,
            hf_split=cfg.hf_split,
            hf_source_field=cfg.hf_source_field,
            hf_source_values=cfg.hf_source_values,
            input_jsonl=cfg.input_jsonl,
            prompt_field=cfg.prompt_field,
            max_prompts=cfg.max_prompts,
            skip_eval_key_substrings=cfg.skip_eval_key_substrings,
        )

        batch: list[tuple[str, str, float]] = []
        target = cfg.target_pairs or 10000

        def flush_batch(batch_items):
            nonlocal n_pairs, n_prompts, n_loops
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.threads) as pool:
                futures = []
                for prompt_text, source, temp in batch_items:
                    fut = pool.submit(
                        _process_single_prompt,
                        client, cfg, model_name, prompt_text, source,
                    )
                    futures.append(fut)
                for fut in concurrent.futures.as_completed(futures):
                    gen_record, pair = fut.result()
                    if gen_record:
                        gen_fh.write(json.dumps(gen_record, ensure_ascii=False) + "\n")
                        n_prompts += 1
                        if gen_record["status"] == "loop_detected":
                            n_loops += 1
                    if pair:
                        pairs_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        n_pairs += 1
                    if n_pairs >= target:
                        return True
            return False

        for prompt_data in prompt_iter:
            text = prompt_data["text"]
            chat = prompt_data["chat_messages"]
            source = prompt_data["source"]

            prompt_text = _format_prompt(text, chat, cfg.prompt_template)
            temp = temperatures[n_prompts % len(temperatures)]
            batch.append((prompt_text, source, temp))

            if len(batch) >= cfg.target_pair_batch_size:
                done = flush_batch(batch)
                batch = []
                logger.info(
                    "progress: prompts=%d, loops=%d, pairs=%d / target=%d",
                    n_prompts, n_loops, n_pairs, target,
                )
                if done or n_pairs >= target:
                    break

        if batch and n_pairs < target:
            flush_batch(batch)

    finally:
        gen_fh.close()
        pairs_fh.close()

    logger.info(
        "generation complete: prompts=%d, loops=%d (%.1f%%), pairs=%d",
        n_prompts, n_loops, (100.0 * n_loops / max(n_prompts, 1)), n_pairs,
    )
    return gen_path, pairs_path, n_pairs
