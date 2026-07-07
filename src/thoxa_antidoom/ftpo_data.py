"""FTPO dataset construction, row sampling, and token regularisation.

Each FTPO row is a single-token preference pair: a prompt prefix ending just
before the loop-starting token, one rejected token (the loop starter), and one
or more chosen tokens (plausible alternatives at that position). This module
handles the tokenization of those rows into model inputs and the pre-training
regularisation that flattens over-represented rejected/chosen tokens.

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from thoxa_antidoom.trimmed import TRIMMED_PROMPT_PREFIX, row_has_trimmed_source_prompt

logger = logging.getLogger(__name__)

DEFAULT_STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "by", "for", "to", "of",
    "and", "or", "but", "if", "then", "else", "when", "where", "how",
    "why", "what", "who", "whom", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "being", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
}

CHOSEN_REGULARISATION_MIN_REF_COUNT = 50
CHOSEN_REGULARISATION_REF_PERCENTILE = 95

SOURCE_BALANCE_MODES = ("off", "sqrt", "equal")


def row_source_label(row: dict[str, Any]) -> str:
    """Extract a human-readable source label from a preference row."""
    return (
        row.get("source_dataset")
        or row.get("source_task")
        or row.get("source_eval_key")
        or "__unknown__"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def has_trimmed_source_prompt(row: dict[str, Any]) -> bool:
    """Check if a row's source prompt was trimmed (and should be discarded)."""
    return row_has_trimmed_source_prompt(row)


def normalise_chosen_key(text: str) -> str:
    """Normalise a chosen token's surface text for deduplication."""
    stripped = text.strip().lower()
    word = re.sub(r"^[^\w]+|[^\w]+$", "", stripped, flags=re.UNICODE)
    return word or stripped or text


def _regularised_ratios(counts: Counter[str], strength: float) -> dict[str, float]:
    """Compute target sampling ratios that flatten over-represented tokens.

    Tokens above the median count get a down-weight of ``(median / count) **
    strength``, pulling their share toward the median. Tokens at or below the
    median are left at their natural proportion.
    """
    if not counts:
        return {}
    median = float(np.median(list(counts.values())))
    weights = {
        tok: 1.0 if count <= median or strength <= 0 else (median / count) ** strength
        for tok, count in counts.items()
    }
    total = sum(weights[tok] * count for tok, count in counts.items())
    return {tok: (weights[tok] * count) / total for tok, count in counts.items()}


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_train_examples: int | None,
    rejected_strength: float,
    min_chosen_tokens: int,
    seed: int,
    source_balance_mode: str = "off",
) -> list[dict[str, Any]]:
    """Sample and regularise FTPO rows before tokenization.

    Applies rejected-token normalisation (flattening over-represented loop
    starters like "Wait" or "Alternatively") and optional source balancing,
    then caps at ``max_train_examples``.
    """
    if source_balance_mode not in SOURCE_BALANCE_MODES:
        raise ValueError(
            f"source_balance_mode must be one of {SOURCE_BALANCE_MODES}; "
            f"got {source_balance_mode!r}"
        )
    rows = [r for r in rows if len(r.get("multi_chosen_decoded") or []) >= min_chosen_tokens]
    if not rows:
        return []
    target_n = min(max_train_examples or len(rows), len(rows))
    selected = _rejected_normalisation(rows, target_n, rejected_strength, seed)
    if source_balance_mode != "off":
        selected = _source_balance_downsample(selected, source_balance_mode, seed)
    return selected


def _rejected_normalisation(
    rows: list[dict[str, Any]],
    target_n: int,
    rejected_strength: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Greedy rejected-token sampling with source-aware tie-breaking.

    The rejected-token target counts are primary; source preference only
    decides which row is taken from a given token bucket when multiple are
    available, biased toward sources currently below their share of the pool.
    """
    ratios = _regularised_ratios(
        Counter(r["rejected_decoded"] for r in rows), rejected_strength
    )
    targets = {tok: int(round(ratio * target_n)) for tok, ratio in ratios.items()}

    pool = Counter(row_source_label(r) for r in rows)
    pool_total = sum(pool.values()) or 1
    share = {s: c / pool_total for s, c in pool.items()}

    buckets: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        buckets[row["rejected_decoded"]][row_source_label(row)].append(row)

    rng = np.random.default_rng(seed)
    for tok in buckets:
        for source in buckets[tok]:
            rng.shuffle(buckets[tok][source])

    selected: list[dict[str, Any]] = []
    seen_tok: defaultdict[str, int] = defaultdict(int)
    selected_src: Counter[str] = Counter()

    def best_choice(respect_cap: bool) -> tuple[str, str] | None:
        denom = max(len(selected), 1)
        best: tuple[str, str] | None = None
        best_key: tuple[float, float] | None = None
        for tok, src_buckets in buckets.items():
            if respect_cap and seen_tok[tok] >= targets.get(tok, 0):
                continue
            tok_gap = ratios.get(tok, 0.0) - seen_tok[tok] / denom
            for source, available in src_buckets.items():
                if not available:
                    continue
                src_def = share.get(source, 0.0) - selected_src[source] / denom
                key = (tok_gap, src_def)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (tok, source)
        return best

    while len(selected) < target_n:
        choice = best_choice(respect_cap=True)
        if choice is None:
            break
        tok, source = choice
        selected.append(buckets[tok][source].pop())
        seen_tok[tok] += 1
        selected_src[source] += 1

    while len(selected) < target_n:
        choice = best_choice(respect_cap=False)
        if choice is None:
            break
        tok, source = choice
        selected.append(buckets[tok][source].pop())
        seen_tok[tok] += 1
        selected_src[source] += 1

    rng.shuffle(selected)
    return selected


def _source_balance_downsample(
    rows: list[dict[str, Any]],
    mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Downsample rows so each source gets at most a sqrt or equal share."""
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row_source_label(row)].append(row)

    n = len(rows)
    n_sources = max(len(by_source), 1)
    if mode == "equal":
        cap = n // n_sources
    elif mode == "sqrt":
        total_weight = sum(len(v) ** 0.5 for v in by_source.values())
        cap = {s: int(n * (len(v) ** 0.5 / total_weight)) for s, v in by_source.items()}
    else:
        return rows

    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    for source, src_rows in by_source.items():
        rng.shuffle(src_rows)
        limit = cap if isinstance(cap, int) else cap.get(source, len(src_rows))
        result.extend(src_rows[:limit])
    rng.shuffle(result)
    return result


class FTPODataset(Dataset):
    """Tokenizes FTPO rows into ``(prompt_ids, chosen_ids, rejected_token_id)``.

    Each row's ``context_before`` (prompt + tokens before the rejection point)
    is tokenized to form ``prompt_ids``. The rejected token and each chosen
    token are tokenized to single token ids. Rows where any chosen token or the
    rejected token do not map to exactly one token id are filtered out.

    Optional ``chosen_regularisation_strength`` downsamples over-represented
    chosen tokens per-row so the adapter learns a broad anti-loop preference
    rather than memorizing one replacement.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        *,
        max_seq_length: int = 4096,
        filter_rejected_stop_words: bool = True,
        chosen_regularisation_strength: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.filter_rejected_stop_words = filter_rejected_stop_words
        self.chosen_regularisation_strength = chosen_regularisation_strength
        self.filter_counts: Counter[str] = Counter()

        self.features: list[dict[str, Any]] = []
        for row in rows:
            feat = self._tokenize_row(row)
            if feat is not None:
                self.features.append(feat)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.features[idx]

    def _tokenize_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        context = row.get("context_before", "")
        if not context:
            self.filter_counts["empty_context"] += 1
            return None

        prompt_ids = self.tokenizer.encode(context, add_special_tokens=True)
        if len(prompt_ids) + 1 > self.max_seq_length:
            self.filter_counts["over_length"] += 1
            return None

        rejected_raw = row.get("rejected_token", "")
        rejected_decoded = row.get("rejected_decoded", "")
        if not rejected_raw:
            self.filter_counts["no_rejected"] += 1
            return None

        rejected_ids = self.tokenizer.encode(rejected_raw, add_special_tokens=False)
        if len(rejected_ids) != 1:
            self.filter_counts["rejected_not_single"] += 1
            return None
        rejected_id = rejected_ids[0]

        if self.filter_rejected_stop_words:
            word = normalise_chosen_key(rejected_decoded)
            if word in DEFAULT_STOP_WORDS:
                self.filter_counts["rejected_stopword"] += 1
                return None

        chosen_raws = row.get("multi_chosen_tokens") or []
        chosen_decoded = row.get("multi_chosen_decoded") or []
        if len(chosen_raws) < 1:
            self.filter_counts["no_chosen"] += 1
            return None

        chosen_ids: list[int] = []
        for raw in chosen_raws:
            ids = self.tokenizer.encode(raw, add_special_tokens=False)
            if len(ids) == 1:
                chosen_ids.append(ids[0])

        if not chosen_ids:
            self.filter_counts["chosen_not_single"] += 1
            return None

        # Chosen regularisation: downsample over-represented chosen tokens.
        if self.chosen_regularisation_strength > 0 and len(chosen_ids) > 1:
            chosen_ids = self._regularise_chosen(chosen_ids, chosen_decoded)

        return {
            "prompt_ids": prompt_ids,
            "chosen_ids": chosen_ids,
            "rejected_token_id": rejected_id,
            "rejected_decoded": rejected_decoded,
            "chosen_decoded": chosen_decoded[: len(chosen_ids)],
        }

    def _regularise_chosen(
        self, chosen_ids: list[int], chosen_decoded: list[str]
    ) -> list[int]:
        """Drop over-represented chosen tokens to balance the chosen pool."""
        if len(chosen_ids) <= 1:
            return chosen_ids
        counts = Counter(normalise_chosen_key(d) for d in chosen_decoded)
        if sum(counts.values()) < CHOSEN_REGULARISATION_MIN_REF_COUNT:
            return chosen_ids
        threshold = float(np.percentile(list(counts.values()), CHOSEN_REGULARISATION_REF_PERCENTILE))
        kept: list[int] = []
        for idx, cid in enumerate(chosen_ids):
            key = normalise_chosen_key(chosen_decoded[idx]) if idx < len(chosen_decoded) else ""
            count = counts.get(key, 0)
            if count <= threshold:
                kept.append(cid)
        return kept if kept else chosen_ids


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
